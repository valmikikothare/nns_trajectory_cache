"""Fuzzy-binning trajectory cache (abstract intermediate class).

A `FuzzyTrajectoryCache` indexes `PlanRequest`s by quantizing every
float (joint angles, goal position, goal orientation) into an integer
bin via `int(value // tolerance)`, then serializing the resulting dict
to JSON bytes. Two requests with the same bin assignments collide on
the same bytes-key — the "fuzzy match" — and share a sorted list of
candidate trajectories.

This class implements the Mapping API of `TrajectoryCache` in terms of
a small set of bytes-keyed storage primitives that concrete subclasses
provide:

- `_get_raw(key)` / `_put_raw(key, value)` — single-entry I/O.
- `_delete_raw(key)` / `_contains_raw(key)` — single-entry presence
  and removal.
- `_clear_storage()` — wipe every entry (called on metadata mismatch).
- `__len__()` — total number of entries (including metadata keys).

Per fuzzy key, up to `max_trajectories` trajectories are kept, sorted
by `path_cost` (length or duration). The most expensive trajectory is
evicted on insert when the cap is exceeded.

See `trajectory_cache_lmdb` and `trajectory_cache_dict` for concrete
backends.
"""

import abc
import bisect
import json
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Literal, Optional

from geometry_msgs.msg import PoseStamped
from moveit.core.robot_state import (  # type: ignore[reportMissingModuleSource]
    RobotState,
)
from moveit.core.robot_trajectory import (  # type: ignore[reportMissingModuleSource]
    RobotTrajectory,
)
from rclpy.impl.rcutils_logger import RcutilsLogger

from tabletop_rig.interfaces.moveit.requests import PlanRequest
from tabletop_rig.interfaces.moveit.trajectory_cache import (
    OrientationToleranceT,
    PositionToleranceT,
    RobotStateToleranceT,
    TrajectoryCache,
    TrajectoryCacheValue,
)
from tabletop_rig.utils.ros import (
    arrays_from_pose_msg,
    get_joint_group_positions,
)

# Metadata keys stored alongside trajectory data. Fuzzy trajectory keys
# are JSON-serialized dicts (always start with '{'), so these plain-word
# keys cannot collide with them.
_META_SCENE_HASH = b"scene_hash"
_META_ROBOT_STATE_TOL = b"robot_state_tolerance"
_META_POSITION_TOL = b"position_tolerance"
_META_ORIENTATION_TOL = b"orientation_tolerance"
_META_MAX_TRAJECTORIES = b"max_trajectories"
_META_PLANNING_FRAME = b"planning_frame"
_META_SORT_BY = b"sort_by"
_META_GROUP_NAME = b"group_name"
_META_POSE_LINK = b"pose_link"


class FuzzyTrajectoryCache(TrajectoryCache):
    """Abstract trajectory cache that bins requests into fuzzy keys.

    Implements `TrajectoryCache`'s Mapping API by quantizing each
    `PlanRequest` into a bytes-key and dispatching to bytes-keyed
    storage primitives that subclasses provide.

    Concurrency: the read-modify-write inside `__setitem__` runs under
    `self._lock` so concurrent threads in a single Python process see
    consistent reads. Cross-process atomicity is the backend's
    responsibility.

    Args:
        (See `TrajectoryCache`. `max_trajectories` is the per-fuzzy-bin
        cap: when an insert pushes the count over the cap, the most
        expensive — highest `path_cost` — entry is evicted.)
    """

    def __init__(
        self,
        *,
        path: str,
        scene_hash: str,
        planning_frame: str,
        group_name: str,
        pose_link: Optional[str] = None,
        robot_state_tolerance: RobotStateToleranceT,
        position_tolerance: PositionToleranceT,
        orientation_tolerance: OrientationToleranceT,
        sort_by: Literal["path_length", "path_duration"] = "path_duration",
        max_trajectories: int = 1,
        parent_logger: Optional[RcutilsLogger] = None,
    ):
        super().__init__(
            path=path,
            scene_hash=scene_hash,
            planning_frame=planning_frame,
            group_name=group_name,
            pose_link=pose_link,
            robot_state_tolerance=robot_state_tolerance,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            sort_by=sort_by,
            max_trajectories=max_trajectories,
            parent_logger=parent_logger,
        )

    # ---------------------------------------------------------------
    # Abstract storage primitives
    # ---------------------------------------------------------------

    @abc.abstractmethod
    def _get_raw(self, key: bytes) -> Any:
        """Return the deserialized value for `key`. Raise KeyError if absent."""

    @abc.abstractmethod
    def _put_raw(self, key: bytes, value: Any) -> None:
        """Store `value` under `key`, serialized however the backend prefers."""

    @abc.abstractmethod
    def _delete_raw(self, key: bytes) -> None:
        """Remove `key`. Raise KeyError if absent."""

    @abc.abstractmethod
    def _contains_raw(self, key: bytes) -> bool:
        """Return True iff `key` is present."""

    @abc.abstractmethod
    def _clear_storage(self) -> None:
        """Wipe every entry in the backend (metadata + trajectories).

        Called when the stored metadata disagrees with the current
        configuration. The cache is open both before and after this
        call; backends that need to close/reopen for the wipe should
        handle that internally.
        """

    # ---------------------------------------------------------------
    # Open-and-validate (called by subclasses at end of __init__)
    # ---------------------------------------------------------------

    def _open_and_validate(self) -> None:
        """Open the backend, validate metadata, then close.

        Subclasses should call this at the end of `__init__` once
        their backend-specific state is set up. The cache is left
        closed; the caller is expected to `open()` (or use the context
        manager) before using it.
        """
        self.open()
        try:
            self._validate_db()
            self.log(
                f"Initialized fuzzy trajectory cache with goal orientation "
                f"tolerance {self._orientation_tolerance}, goal position "
                f"tolerance {self._position_tolerance}, robot state "
                f"tolerance {self._robot_state_tolerance}, and max "
                f"trajectories {self._max_trajectories}."
            )
        finally:
            self.close()

    def _validate_db(self) -> None:
        """Validate stored metadata against the current configuration.

        If any metadata key is missing or has the wrong value, the
        backend is wiped via `_clear_storage` and rewritten with
        fresh metadata.
        """
        metadata: dict[bytes, Any] = {
            _META_SCENE_HASH: self._scene_hash,
            _META_PLANNING_FRAME: self._planning_frame,
            _META_GROUP_NAME: self._group_name,
            _META_POSE_LINK: self._pose_link,
            _META_ROBOT_STATE_TOL: deepcopy(self._robot_state_tolerance),
            _META_POSITION_TOL: self._position_tolerance,
            _META_ORIENTATION_TOL: self._orientation_tolerance,
            _META_SORT_BY: self._sort_by,
            _META_MAX_TRAJECTORIES: self._max_trajectories,
        }

        if len(self) > 0:
            mismatch = False
            for key, value in metadata.items():
                try:
                    old_value = self._get_raw(key)
                except KeyError:
                    mismatch = True
                    self.log(
                        f"Cache is not empty, but key {key!r} is missing.",
                        severity="WARN",
                    )
                    continue
                if old_value != value:
                    mismatch = True
                    self.log(
                        f"Old {key!r} value in db is different from new "
                        f"value: {old_value} != {value}.",
                        severity="WARN",
                    )

            if not mismatch:
                return

            self.log(
                "Wiping existing cache contents and recreating...",
                severity="WARN",
            )
            self._clear_storage()

        for key, value in metadata.items():
            self._put_raw(key, value)

    # ---------------------------------------------------------------
    # Mapping API (concrete; uses fuzzy keys + storage primitives)
    # ---------------------------------------------------------------

    def __setitem__(
        self, request: PlanRequest, trajectory: RobotTrajectory
    ) -> None:
        """Insert `trajectory` under the fuzzy key for `request`.

        Maintains at most `max_trajectories` per fuzzy key, evicting
        the most expensive trajectory when over the cap. The full
        read-modify-write is held under `self._lock` so concurrent
        Python threads cannot interleave.
        """
        self._validate_request(request)
        fuzzy_key = self._fuzzy_key_bytes(request)
        value = TrajectoryCacheValue(trajectory, self._sort_by)
        self.log(f"Setting item for key: {fuzzy_key!r}", severity="DEBUG")

        with self._lock:
            try:
                values = self._get_raw(fuzzy_key)
            except KeyError:
                values: list[TrajectoryCacheValue] = []
            else:
                self._validate_db_values(values)

            bisect.insort_left(values, value)
            if len(values) > self._max_trajectories:
                values.pop(-1)

            self._put_raw(fuzzy_key, values)

    def __getitem__(self, request: PlanRequest) -> list[RobotTrajectory]:
        """Return matching trajectories for `request`, ranked best-first."""
        self._validate_request(request)
        fuzzy_key = self._fuzzy_key_bytes(request)
        self.log(f"Getting values for key: {fuzzy_key!r}", severity="DEBUG")

        values = self._get_raw(fuzzy_key)
        self._validate_db_values(values)
        return [v.get_trajectory(request.start_state) for v in values]

    def __contains__(self, request: PlanRequest) -> bool:
        """Check if `request`'s fuzzy bin has any cached trajectories."""
        self._validate_request(request)
        return self._contains_raw(self._fuzzy_key_bytes(request))

    def __delitem__(self, request: PlanRequest) -> None:
        """Delete every trajectory in `request`'s fuzzy bin."""
        self._validate_request(request)
        self._delete_raw(self._fuzzy_key_bytes(request))

    def _validate_db_values(self, values: list[TrajectoryCacheValue]) -> None:
        """Sanity-check a list of values pulled from the backend."""
        if not __debug__:
            return

        assert isinstance(values, list)
        assert all(isinstance(v, TrajectoryCacheValue) for v in values)
        assert 1 <= len(values) <= self._max_trajectories

    # ---------------------------------------------------------------
    # Fuzzy-key construction
    # ---------------------------------------------------------------

    def _fuzzy_key_bytes(self, request: PlanRequest) -> bytes:
        """Compute the fuzzy bytes-key for `request`."""
        return json.dumps(
            self._fuzzy_key_dict(request), sort_keys=True
        ).encode("utf-8")

    def _fuzzy_key_dict(self, request: PlanRequest) -> dict[str, Any]:
        """Compute the fuzzy key as a dict.

        Joint angles and Cartesian coordinates are each quantized into
        integer bins via the configured tolerances. JSON-serializing
        this dict yields the bytes-key used by the storage layer.

        `group_name`, `pose_link`, and the Cartesian goal's `frame_id`
        are intentionally absent — they are stored once as cache-level
        metadata and validated against on every request, so they need
        not bloat every fuzzy bin's key.
        """
        start_state = request.start_state
        goal = request.goal

        assert start_state is not None

        positions = get_joint_group_positions(start_state, self._group_name)
        fuzzy: dict[str, Any] = {
            "start_state": self._fuzz_dict(
                positions, self._robot_state_tolerance
            ),
        }

        if isinstance(goal, RobotState):
            goal_positions = get_joint_group_positions(goal, self._group_name)
            fuzzy["goal_joints"] = self._fuzz_dict(
                goal_positions, self._robot_state_tolerance
            )
        else:
            assert isinstance(goal, PoseStamped)
            goal_position, goal_orientation = arrays_from_pose_msg(
                goal.pose, euler=False
            )
            fuzzy["goal_pose"] = {
                "position": self._fuzz_iterable(
                    goal_position, self._position_tolerance
                ),
                "orientation": self._fuzz_iterable(
                    goal_orientation, self._orientation_tolerance
                ),
            }

        return fuzzy

    @staticmethod
    def _fuzz_float(value: float, tolerance: float) -> int:
        return int(value // tolerance)

    @classmethod
    def _fuzz_iterable(
        cls,
        value: Iterable[float],
        tolerance: float | Iterable[float],
    ) -> tuple[int, ...]:
        if isinstance(tolerance, (float, int)):
            return tuple(cls._fuzz_float(v, tolerance) for v in value)
        return tuple(cls._fuzz_float(v, t) for v, t in zip(value, tolerance))

    @classmethod
    def _fuzz_dict(
        cls,
        value: dict[str, float],
        tolerance: float | dict[str, float],
    ) -> dict[str, int]:
        if isinstance(tolerance, (float, int)):
            return {k: cls._fuzz_float(v, tolerance) for k, v in value.items()}
        return {k: cls._fuzz_float(value[k], tolerance[k]) for k in value}
