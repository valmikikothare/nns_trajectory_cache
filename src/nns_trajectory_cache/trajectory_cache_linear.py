"""Linear-search trajectory cache (in-memory brute-force baseline).

A `LinearTrajectoryCache` stores every cached request under an exact
(non-fuzzy) fingerprint and answers queries by linearly scanning every
stored fingerprint, returning those within the configured tolerances.
It is the brute-force baseline against which approximate-nearest-
neighbor backends (fuzzy binning, LSH, k-d trees) are compared.

Insert is O(1) amortized; query is O(N) in the number of stored
fingerprints. The dict structure mirrors `DictFuzzyTrajectoryCache` so
both backends have comparable per-operation overhead aside from the
lookup-strategy difference itself.

The component-wise tolerance check `_close_iterable` / `_close_dict`
mirrors the equivalence relation `FuzzyTrajectoryCache` induces via
`int(x // tolerance)` binning — two requests that would land in the
same fuzzy bin always satisfy the linear cache's tolerance check (and
vice versa, up to the bin-boundary edge case). This keeps the two
backends comparable on identical input.
"""

import bisect
import json
import os
import pickle
from collections.abc import Iterable
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

_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


class LinearTrajectoryCache(TrajectoryCache):
    """Brute-force linear-scan trajectory cache (in-memory).

    Each unique exact-fingerprint request is one dict entry holding up
    to `max_trajectories` trajectories sorted by `path_cost`. Lookup
    iterates every stored entry, runs a component-wise tolerance check
    against the query, collects all matching trajectories, sorts them
    by cost, and returns the cheapest `max_trajectories` of them.

    Insert deduplicates on byte-identical fingerprints (so re-caching
    the same trajectory is a no-op modulo path-cost re-sorting); near-
    but-not-identical requests are stored as separate entries, which
    is what makes the storage grow linearly with insert calls in
    contrast to fuzzy's bin-collapsing behavior.

    Args:
        path: Absolute path to a pickle file. If provided, the cache
            is loaded from this file on `open()` and saved on
            `close()`. If `None`, the cache is purely process-local.
        (See `TrajectoryCache`. `max_trajectories` is the per-entry
        cap on insert *and* the cap on results returned per query.)
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

        # Entries keyed by an exact JSON serialization of the request
        # fingerprint (so byte-identical requests dedup). Each value
        # is the parsed fingerprint dict plus a cost-sorted list of
        # at most `max_trajectories` candidates.
        self._store: dict[
            bytes, tuple[dict[str, Any], list[TrajectoryCacheValue]]
        ] = {}

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Trajectory cache is not open")

    # ---------------------------------------------------------------
    # Mapping API
    # ---------------------------------------------------------------

    def __setitem__(
        self, request: PlanRequest, trajectory: RobotTrajectory
    ) -> None:
        """Insert `trajectory` under the exact key for `request`.

        Maintains at most `max_trajectories` per exact key, evicting
        the most expensive trajectory when over the cap. The read-
        modify-write is held under `self._lock`.
        """
        self._validate_request(request)
        self._require_open()
        exact_dict = self._exact_key_dict(request)
        exact_bytes = self._exact_key_bytes(exact_dict)
        value = TrajectoryCacheValue(trajectory, self._sort_by)
        self.log(f"Setting item for key: {exact_bytes!r}", severity="DEBUG")

        with self._lock:
            entry = self._store.get(exact_bytes)
            if entry is None:
                values: list[TrajectoryCacheValue] = []
            else:
                values = entry[1]

            bisect.insort_left(values, value)
            if len(values) > self._max_trajectories:
                values.pop(-1)

            self._store[exact_bytes] = (exact_dict, values)

    def __getitem__(self, request: PlanRequest) -> list[RobotTrajectory]:
        """Return matching trajectories for `request`, ranked best-first.

        Linearly scans every stored entry, collecting candidates whose
        stored fingerprint is within tolerance of the query's. The
        full pool of candidates is then sorted by `path_cost` and
        capped at `max_trajectories`.
        """
        self._validate_request(request)
        self._require_open()
        query_dict = self._exact_key_dict(request)

        all_matches: list[TrajectoryCacheValue] = []
        for stored_dict, values in self._store.values():
            if self._within_tolerance(query_dict, stored_dict):
                all_matches.extend(values)

        if not all_matches:
            raise KeyError(request)

        all_matches.sort()
        capped = all_matches[: self._max_trajectories]
        assert request.start_state is not None
        return [v.get_trajectory(request.start_state) for v in capped]

    def __contains__(self, request: PlanRequest) -> bool:
        """Check if any stored entry is within tolerance of `request`."""
        self._validate_request(request)
        self._require_open()
        query_dict = self._exact_key_dict(request)
        for stored_dict, _ in self._store.values():
            if self._within_tolerance(query_dict, stored_dict):
                return True
        return False

    def __delitem__(self, request: PlanRequest) -> None:
        """Delete every entry within tolerance of `request`.

        Raises:
            KeyError: If no entry matches.
        """
        self._validate_request(request)
        self._require_open()
        query_dict = self._exact_key_dict(request)

        with self._lock:
            to_remove = [
                key
                for key, (stored_dict, _) in self._store.items()
                if self._within_tolerance(query_dict, stored_dict)
            ]
            if not to_remove:
                raise KeyError(request)
            for key in to_remove:
                del self._store[key]

    def __len__(self) -> int:
        """Total number of stored entries (distinct exact fingerprints)."""
        self._require_open()
        return len(self._store)

    def open(self) -> None:
        """Mark the cache as open and, if `path` is set, load from disk.

        A missing or unreadable file is treated as "start fresh" — the
        store stays empty and execution continues.
        """
        with self._lock:
            if not self._closed:
                self.log("Cache is already open", severity="WARN")
                return
            if os.path.exists(self._path):
                try:
                    with open(self._path, "rb") as f:
                        self._store = pickle.load(f)
                    self.log(
                        f"Loaded {len(self._store)} entries from {self._path}",
                        severity="INFO",
                    )
                except Exception as e:
                    self.log(
                        f"Failed to load cache from {self._path}: {e}. "
                        f"Starting fresh.",
                        severity="WARN",
                    )
                    self._store = {}
            self._closed = False

    def close(self) -> None:
        """Mark the cache as closed and, if `path` is set, save to disk."""
        with self._lock:
            if self._closed:
                self.log("Cache is already closed", severity="WARN")
                return
            try:
                with open(self._path, "wb") as f:
                    pickle.dump(self._store, f, protocol=_PICKLE_PROTOCOL)
                self.log(
                    f"Saved {len(self._store)} entries to {self._path}",
                    severity="INFO",
                )
            except Exception as e:
                self.log(
                    f"Failed to save cache to {self._path}: {e}",
                    severity="ERROR",
                )
            self._closed = True

    # ---------------------------------------------------------------
    # Exact-key construction
    # ---------------------------------------------------------------

    def _exact_key_dict(self, request: PlanRequest) -> dict[str, Any]:
        """Compute the exact (non-fuzzy) fingerprint dict for `request`.

        Mirrors `FuzzyTrajectoryCache._fuzzy_key_dict` but skips the
        integer-quantization step — all floats are kept at full
        precision. Tolerance comparison happens later, on lookup.

        `group_name`, `pose_link`, and the Cartesian goal's `frame_id`
        are stored as cache-level metadata and validated on every
        request, so they're absent from the fingerprint. Joint-space
        goals are encoded under `"goal_joints"`; Cartesian goals under
        `"goal_pose"`. `_within_tolerance` discriminates the two via
        which key is present.
        """
        start_state = request.start_state
        goal = request.goal

        assert start_state is not None

        positions = get_joint_group_positions(start_state, self._group_name)
        exact: dict[str, Any] = {
            "start_state": dict(positions),
        }

        if isinstance(goal, RobotState):
            goal_positions = get_joint_group_positions(goal, self._group_name)
            exact["goal_joints"] = dict(goal_positions)
        else:
            assert isinstance(goal, PoseStamped)
            goal_position, goal_orientation = arrays_from_pose_msg(
                goal.pose, euler=False
            )
            exact["goal_pose"] = {
                "position": tuple(goal_position),
                "orientation": tuple(goal_orientation),
            }

        return exact

    @staticmethod
    def _exact_key_bytes(exact_dict: dict[str, Any]) -> bytes:
        """Canonical bytes encoding used as the dict-store key.

        Used only to dedup byte-identical insertions; the tolerance
        check on `__getitem__` reads the parsed dict directly.
        """
        return json.dumps(exact_dict, sort_keys=True).encode("utf-8")

    # ---------------------------------------------------------------
    # Tolerance comparison
    # ---------------------------------------------------------------

    def _within_tolerance(
        self, query: dict[str, Any], stored: dict[str, Any]
    ) -> bool:
        """Return True iff `stored`'s fingerprint matches `query`'s.

        `group_name`, `pose_link`, and Cartesian `frame_id` are already
        enforced equal by `_validate_request`, so the tolerance check
        here only compares the float features:
          - Start-state joint positions within `robot_state_tolerance`.
          - The goal types (joint-space vs Cartesian) must match —
            the fingerprint stores them under disjoint keys
            (`goal_joints` vs `goal_pose`).
          - Joint-space goals: per-joint within `robot_state_tolerance`.
          - Cartesian goals: position components within
            `position_tolerance`, orientation components within
            `orientation_tolerance`.
        """
        if not self._close_dict(
            query["start_state"],
            stored["start_state"],
            self._robot_state_tolerance,
        ):
            return False

        q_is_pose = "goal_pose" in query
        s_is_pose = "goal_pose" in stored
        if q_is_pose != s_is_pose:
            return False

        if q_is_pose:
            q_goal = query["goal_pose"]
            s_goal = stored["goal_pose"]
            if not self._close_iterable(
                q_goal["position"],
                s_goal["position"],
                self._position_tolerance,
            ):
                return False
            if not self._close_iterable(
                q_goal["orientation"],
                s_goal["orientation"],
                self._orientation_tolerance,
            ):
                return False
        else:
            if not self._close_dict(
                query["goal_joints"],
                stored["goal_joints"],
                self._robot_state_tolerance,
            ):
                return False

        return True

    @staticmethod
    def _close_iterable(
        a: Iterable[float],
        b: Iterable[float],
        tolerance: float | Iterable[float],
    ) -> bool:
        if isinstance(tolerance, (int, float)):
            return all(abs(x - y) <= tolerance for x, y in zip(a, b))
        return all(abs(x - y) <= t for x, y, t in zip(a, b, tolerance))

    @staticmethod
    def _close_dict(
        a: dict[str, float],
        b: dict[str, float],
        tolerance: float | dict[str, float],
    ) -> bool:
        if a.keys() != b.keys():
            return False
        if isinstance(tolerance, (int, float)):
            return all(abs(a[k] - b[k]) <= tolerance for k in a)
        return all(abs(a[k] - b[k]) <= tolerance[k] for k in a)
