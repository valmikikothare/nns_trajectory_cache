"""Trajectory caching for motion planning (abstract base).

This module defines the storage-agnostic abstract base class
`TrajectoryCache`, which exposes a `Mapping`-like API keyed on
`PlanRequest` and valued on `RobotTrajectory`. The base class does not
know how requests are indexed — that's the job of concrete backends.

Concrete backends should subclass `TrajectoryCache` and implement the
Mapping primitives (`__setitem__`, `__getitem__`, `__contains__`,
`__delitem__`) plus the lifecycle hooks (`open`, `close`, `__len__`).

A fuzzy-binning intermediate class (`FuzzyTrajectoryCache`) lives in
`trajectory_cache_fuzzy`; LSH or k-d tree backends would subclass
`TrajectoryCache` directly.

Classes:
    TrajectoryCache: Abstract base class for trajectory caches keyed on
        PlanRequest.
"""

import abc
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Optional

import rclpy.logging
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_state import (  # type: ignore[reportMissingModuleSource]
    RobotState,
)
from moveit.core.robot_trajectory import (  # type: ignore[reportMissingModuleSource]
    RobotTrajectory,
)
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from rclpy.impl.rcutils_logger import RcutilsLogger

from tabletop_py.utils.common import is_iterable
from tabletop_rig.interfaces.moveit.requests import PlanRequest
from tabletop_rig.utils.logging import LoggerMixin
from tabletop_rig.utils.ros import (
    all_close_poses_stamped,
    all_close_robot_states,
    get_joint_group_positions,
    pose_stamped_msg,
    robot_trajectory_from_msg,
)

RobotStateToleranceT = float | dict[str, float]
PositionToleranceT = float | tuple[float, float, float]
OrientationToleranceT = (
    float | tuple[float, float, float] | tuple[float, float, float, float]
)


@dataclass(slots=True, frozen=True, eq=False)
class TrajectoryCacheValue:
    """A trajectory wrapped with its sortable path cost.

    Used by `TrajectoryCache` subclasses to keep cached trajectories
    ranked by `path_cost` (lower is better). The trajectory is held in
    its msg form so it's portable across robot-state contexts;
    `get_trajectory(state)` rehydrates it bound to a given start state
    before it's handed back through the public Mapping API.
    """

    trajectory_msg: RobotTrajectoryMsg
    group_name: str
    path_cost: float

    def __init__(
        self,
        trajectory: RobotTrajectory,
        sort_by: Literal["path_length", "path_duration"],
    ):
        if not isinstance(trajectory, RobotTrajectory):
            raise ValueError(
                f"Trajectory is not a RobotTrajectory: {trajectory}"
            )
        object.__setattr__(
            self, "trajectory_msg", trajectory.get_robot_trajectory_msg()
        )
        object.__setattr__(
            self, "group_name", trajectory.joint_model_group_name
        )
        if sort_by == "path_length":
            object.__setattr__(self, "path_cost", trajectory.path_length)
        elif sort_by == "path_duration":
            if len(trajectory) > 1 and trajectory.duration <= 0:
                raise ValueError(
                    "If 'sort_by' is set to 'path_duration', the trajectory "
                    "must have a duration (can be set by performing TOTG on "
                    "the trajectory before caching)"
                )
            object.__setattr__(self, "path_cost", trajectory.duration)
        else:
            raise ValueError(
                "'sort_by' must be one of 'path_length' or 'path_duration'"
            )

    def get_trajectory(self, state: RobotState) -> RobotTrajectory:
        return robot_trajectory_from_msg(
            self.trajectory_msg, state, self.group_name
        )

    def __lt__(self, other: "TrajectoryCacheValue") -> bool:
        return self.path_cost < other.path_cost


class TrajectoryCache(LoggerMixin, metaclass=abc.ABCMeta):
    """Abstract base for trajectory caches keyed on `PlanRequest`.

    Subclasses provide the Mapping primitives (`__setitem__`,
    `__getitem__`, `__contains__`, `__delitem__`) and the lifecycle
    hooks (`open`, `close`, `__len__`). Everything else — high-level
    `cache_trajectory` bookkeeping, request and trajectory-quality
    validation, tolerance management — lives here.

    The Mapping API treats `PlanRequest` as the key (start state +
    goal + group name + optional pose link) and `RobotTrajectory` as
    the value. `__getitem__` returns a list of trajectories ranked
    best-first (cheapest path cost first); subclasses are responsible
    for honoring `sort_by` when constructing that ranking.

    Args:
        scene_hash: Hash describing the static scene/rig configuration.
            Subclasses may use it to detect stale persistent state and
            wipe it.
        planning_frame: The MoveIt planning frame. All cached requests
            and trajectories must live in this frame.
        group_name: The joint model group every request must address.
            Stored once as cache metadata rather than per-key, since
            it never varies for a given cache instance.
        pose_link: The end-effector link Cartesian goals are expressed
            against. Like `group_name`, stored once as metadata. May
            be `None`, in which case the cache only accepts joint-
            space (`RobotState`) goal requests.
        robot_state_tolerance: Per-joint angle tolerance. Used by
            `_validate_trajectory_quality` to confirm cached trajectory
            endpoints agree with the request, and by fuzzy backends
            for binning.
        position_tolerance: Cartesian goal position tolerance.
        orientation_tolerance: Cartesian goal orientation tolerance.
        sort_by: Whether to rank cached trajectories by `path_length`
            or `path_duration`.
        max_trajectories: Cap on cached trajectories kept per match
            group. Subclasses define what a "match group" means
            (per-bin for fuzzy backends, per-tolerance-cluster for the
            linear baseline, etc.).
        parent_logger: Optional ROS logger to derive the cache logger
            from.
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
        if parent_logger is None:
            self._logger = rclpy.logging.get_logger("trajectory_cache")
        else:
            self._logger = parent_logger.get_child("trajectory_cache")

        self._path = self._normalize_path(path)

        if sort_by not in ("path_length", "path_duration"):
            raise ValueError(
                "'sort_by' must be one of 'path_length' or 'path_duration'"
            )
        self._sort_by: Literal["path_length", "path_duration"] = sort_by

        if max_trajectories < 1:
            raise ValueError("'max_trajectories' must be at least 1")
        self._max_trajectories = max_trajectories

        if not isinstance(group_name, str) or not group_name:
            raise TypeError(
                f"'group_name' must be a non-empty string: {group_name!r}"
            )
        self._group_name = group_name

        if pose_link is not None and (
            not isinstance(pose_link, str) or not pose_link
        ):
            raise TypeError(
                f"'pose_link' must be None or a non-empty string: "
                f"{pose_link!r}"
            )
        self._pose_link = pose_link

        self._planning_frame = planning_frame
        self._scene_hash = scene_hash

        (
            self._robot_state_tolerance,
            self._position_tolerance,
            self._orientation_tolerance,
        ) = self._init_tolerances(
            robot_state_tolerance,
            position_tolerance,
            orientation_tolerance,
        )

        # Available to subclasses for serializing read-modify-write
        # operations on shared state.
        self._lock = threading.Lock()
        self._closed = True

    def get_logger(self) -> RcutilsLogger:
        """Get the logger instance."""
        return self._logger

    @property
    def path(self) -> str:
        """The path to the database env."""
        return self._path

    @staticmethod
    def _init_tolerances(
        robot_state_tolerance: Any,
        position_tolerance: Any,
        orientation_tolerance: Any,
    ) -> tuple[
        RobotStateToleranceT, PositionToleranceT, OrientationToleranceT
    ]:
        """Validate and normalize the tolerance parameters."""
        if isinstance(robot_state_tolerance, Mapping):
            robot_state_tolerance = {
                k: float(v) for k, v in robot_state_tolerance.items()
            }
            if len(robot_state_tolerance) != 6:
                raise ValueError("robot_state_tolerance must be a 6-tuple")
            if any(x <= 0 for x in robot_state_tolerance.values()):
                raise ValueError("robot_state_tolerance must be positive")
        elif robot_state_tolerance <= 0:
            raise ValueError("robot_state_tolerance must be positive")

        if is_iterable(position_tolerance):
            position_tolerance = tuple(map(float, position_tolerance))
            if len(position_tolerance) != 3:
                raise ValueError("position_tolerance must be a 3-tuple")
            if any(x <= 0 for x in position_tolerance):
                raise ValueError("position_tolerance must be positive")
        elif position_tolerance <= 0:
            raise ValueError("position_tolerance must be positive")

        if is_iterable(orientation_tolerance):
            orientation_tolerance = tuple(map(float, orientation_tolerance))
            if len(orientation_tolerance) != 4:
                raise ValueError(
                    f"orientation_tolerance must be a 4-tuple "
                    f"but got a {len(orientation_tolerance)}-tuple"
                )
            if any(x <= 0 for x in orientation_tolerance):
                raise ValueError("orientation_tolerance must be positive")
        elif orientation_tolerance <= 0:
            raise ValueError("orientation_tolerance must be positive")

        return (
            robot_state_tolerance,
            position_tolerance,
            orientation_tolerance,
        )

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize and validate a persistence file path.

        Returns the absolute path with `~` and `$VAR`s expanded, after
        ensuring the parent directory exists.

        Raises:
            ValueError: If `path` is relative or names something that
                already exists and is not a regular file.
        """
        path = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isabs(path):
            raise ValueError(f"Trajectory cache path must be absolute: {path}")
        if os.path.exists(path) and not os.path.isfile(path):
            raise ValueError(
                f"Trajectory cache path must be a regular file: {path}"
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------

    @property
    def scene_hash(self) -> str:
        """Scene hash this cache was configured with."""
        return self._scene_hash

    @property
    def planning_frame(self) -> str:
        """The planning frame."""
        return self._planning_frame

    @property
    def group_name(self) -> str:
        """The joint model group this cache is configured for."""
        return self._group_name

    @property
    def pose_link(self) -> Optional[str]:
        """The end-effector link Cartesian goals are expressed against.

        `None` means this cache rejects PoseStamped goals entirely.
        """
        return self._pose_link

    @property
    def robot_state_tolerance(self) -> RobotStateToleranceT:
        """Per-joint angle tolerance."""
        return self._robot_state_tolerance

    @property
    def position_tolerance(self) -> PositionToleranceT:
        """Cartesian goal position tolerance."""
        return self._position_tolerance

    @property
    def orientation_tolerance(self) -> OrientationToleranceT:
        """Cartesian goal orientation tolerance."""
        return self._orientation_tolerance

    @property
    def sort_by(self) -> Literal["path_length", "path_duration"]:
        """How cached trajectories are ranked."""
        return self._sort_by

    @property
    def max_trajectories(self) -> int:
        """Max cached trajectories kept per match group."""
        return self._max_trajectories

    # ---------------------------------------------------------------
    # Abstract Mapping-like API
    # ---------------------------------------------------------------

    @abc.abstractmethod
    def __setitem__(
        self, request: PlanRequest, trajectory: RobotTrajectory
    ) -> None:
        """Insert a trajectory under the given request.

        Subclasses define what "insert" means — overwriting, appending
        to a ranked list, eviction policy, etc.
        """

    @abc.abstractmethod
    def __getitem__(self, request: PlanRequest) -> list[RobotTrajectory]:
        """Return matching trajectories for the request, ranked best-first.

        Raises:
            KeyError: If no matching entry is found.
        """

    @abc.abstractmethod
    def __contains__(self, request: PlanRequest) -> bool:
        """Return True iff at least one matching trajectory exists."""

    @abc.abstractmethod
    def __delitem__(self, request: PlanRequest) -> None:
        """Delete all matching trajectories for the request.

        Raises:
            KeyError: If no matching entry is found.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Total number of entries (definition is implementation-specific)."""

    @abc.abstractmethod
    def open(self) -> None:
        """Open the backend so reads and writes can proceed."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release backend resources."""

    # ---------------------------------------------------------------
    # Request / trajectory validation
    # ---------------------------------------------------------------

    def _validate_request(self, request: PlanRequest) -> None:
        """Validate that `request` is well-formed for cache I/O.

        Subclasses may rely on these invariants when implementing the
        Mapping API; the high-level methods on this class also call
        it before constructing synthetic requests.
        """
        start_state = request.start_state
        goal = request.goal
        pose_link = request.pose_link
        group_name = request.group_name

        if not isinstance(start_state, RobotState):
            raise TypeError(f"Start state must be a RobotState: {start_state}")
        if not isinstance(goal, (RobotState, PoseStamped)):
            raise TypeError(
                f"Goal must be a RobotState or PoseStamped (named-target "
                f"goals are not cacheable): {goal}"
            )
        if pose_link is not None and not isinstance(pose_link, str):
            raise TypeError(f"Pose link must be a string: {pose_link}")
        if not isinstance(group_name, str):
            raise TypeError(f"Group name must be a string: {group_name}")

        if group_name != self._group_name:
            raise ValueError(
                f"Request group_name {group_name!r} does not match the "
                f"cache's configured group_name {self._group_name!r}"
            )

        if start_state.robot_model.model_frame != self._planning_frame:
            raise ValueError(
                f"Start state robot model frame must be "
                f"'{self._planning_frame}': "
                f"{start_state.robot_model.model_frame}"
            )

        if not start_state.robot_model.has_joint_model_group(group_name):
            raise ValueError(
                f"Start state robot model must have joint model group: "
                f"{group_name}"
            )

        if isinstance(goal, RobotState):
            if goal.robot_model.model_frame != self._planning_frame:
                raise ValueError(
                    f"Goal robot model frame must be "
                    f"'{self._planning_frame}': "
                    f"{goal.robot_model.model_frame}"
                )
            if pose_link is not None:
                raise ValueError(
                    f"Pose link must not be provided for a RobotState "
                    f"goal: {pose_link}"
                )
            if not goal.robot_model.has_joint_model_group(group_name):
                raise ValueError(
                    f"Goal robot model must have joint model group: "
                    f"{group_name}"
                )
        else:
            if self._pose_link is None:
                raise ValueError(
                    "Cache was configured without a 'pose_link'; "
                    "PoseStamped goals are not accepted"
                )
            if goal.header.frame_id != self._planning_frame:
                raise ValueError(
                    f"Goal pose frame id must be "
                    f"'{self._planning_frame}': {goal.header.frame_id}"
                )
            if pose_link is None:
                raise ValueError(
                    "Pose link must be provided for a PoseStamped goal"
                )
            if pose_link != self._pose_link:
                raise ValueError(
                    f"Request pose_link {pose_link!r} does not match the "
                    f"cache's configured pose_link {self._pose_link!r}"
                )

    def _validate_trajectory_quality(
        self, trajectory: RobotTrajectory, request: PlanRequest
    ) -> None:
        """Check that `trajectory`'s endpoints agree with `request`.

        The trajectory's first state must be close to `request.start_state`
        (per `robot_state_tolerance`). The trajectory's last state must
        be close to `request.goal` — either pose-wise (if the goal is a
        PoseStamped) or joint-wise (if the goal is a RobotState).

        Assumes `request` has already passed `_validate_request`.
        """
        # Narrow types for the type checker (validated by _validate_request).
        assert isinstance(request.start_state, RobotState)
        assert isinstance(request.goal, (RobotState, PoseStamped))

        group_name = trajectory.joint_model_group_name
        trajectory_start_state: RobotState = trajectory[0]
        trajectory_end_state: RobotState = trajectory[len(trajectory) - 1]

        if not all_close_robot_states(
            trajectory_start_state,
            request.start_state,
            group_name=group_name,
            position_tolerance=self.robot_state_tolerance,
        ):
            raise ValueError(
                "Request start state is not close to the trajectory start "
                f"state. Request start joint positions: "
                f"{get_joint_group_positions(request.start_state, group_name)}, "
                f"Trajectory start joint positions: "
                f"{get_joint_group_positions(trajectory_start_state, group_name)}"
            )

        trajectory_end_pose = None
        if request.pose_link is not None:
            trajectory_start_pose = pose_stamped_msg(
                pose=trajectory_start_state.get_pose(request.pose_link),
                frame_id=trajectory_start_state.robot_model.model_frame,
            )
            trajectory_end_pose = pose_stamped_msg(
                pose=trajectory_end_state.get_pose(request.pose_link),
                frame_id=trajectory_end_state.robot_model.model_frame,
            )

            request_start_pose = pose_stamped_msg(
                pose=request.start_state.get_pose(request.pose_link),
                frame_id=request.start_state.robot_model.model_frame,
            )
            if not all_close_poses_stamped(
                trajectory_start_pose,
                request_start_pose,
                position_tolerance=self.position_tolerance,
                orientation_tolerance=self.orientation_tolerance,
            ):
                raise ValueError(
                    "Request start pose is not close to the trajectory start "
                    f"pose. Request start pose: {request_start_pose}, "
                    f"Trajectory start pose: {trajectory_start_pose}"
                )

        if isinstance(request.goal, RobotState):
            if not all_close_robot_states(
                trajectory_end_state,
                request.goal,
                group_name=group_name,
                position_tolerance=self.robot_state_tolerance,
            ):
                raise ValueError(
                    "Request goal state is not close to the trajectory end "
                    f"state. Request goal joint positions: "
                    f"{get_joint_group_positions(request.goal, group_name)}, "
                    f"Trajectory end joint positions: "
                    f"{get_joint_group_positions(trajectory_end_state, group_name)}"
                )
        else:
            assert trajectory_end_pose is not None
            if not all_close_poses_stamped(
                trajectory_end_pose,
                request.goal,
                position_tolerance=self.position_tolerance,
                orientation_tolerance=self.orientation_tolerance,
            ):
                raise ValueError(
                    "Request goal pose is not close to the trajectory end "
                    f"pose. Request goal pose: {request.goal}, "
                    f"Trajectory end pose: {trajectory_end_pose}"
                )

    # ---------------------------------------------------------------
    # High-level API (concrete, in terms of Mapping)
    # ---------------------------------------------------------------

    def cache_trajectory(
        self,
        trajectory: RobotTrajectory,
        *,
        request: PlanRequest,
        validate: bool = True,
        _reverse: bool = True,
    ) -> None:
        """Cache `trajectory`, indexed by both joint-space and Cartesian goals.

        The trajectory is inserted under two synthetic `PlanRequest`s:
        one whose goal is the trajectory's end `RobotState`, and one
        whose goal is the trajectory's end Cartesian pose. This lets
        future lookups via either kind of goal hit the cache.

        With `_reverse=True`, the time-reversed trajectory is also
        cached so motions in either direction can be reused.

        Args:
            trajectory: The trajectory to cache. Its
                `joint_model_group_name` must equal the cache's
                configured `group_name`.
            request: The original planning request (used for
                trajectory-quality validation). The synthetic
                requests written to the cache use the cache's own
                `group_name` and `pose_link`, not the request's.
            validate: If True, verify that the trajectory's endpoints
                match the request before caching.
            _reverse: If True, also cache the time-reversed trajectory.
                Internal recursion flag; do not set from outside.
        """
        self._validate_request(request)

        if trajectory.joint_model_group_name != self._group_name:
            raise ValueError(
                f"Trajectory's group_name "
                f"{trajectory.joint_model_group_name!r} does not match "
                f"the cache's configured group_name {self._group_name!r}"
            )

        if validate and _reverse:
            try:
                self._validate_trajectory_quality(trajectory, request)
            except ValueError as e:
                self.log(
                    f"Trajectory is not valid: {e}. Skipping cache.",
                    severity="WARN",
                )
                raise

        start_state: RobotState = trajectory[0]
        end_state: RobotState = trajectory[len(trajectory) - 1]

        state_request = PlanRequest(
            start_state=start_state,
            goal=end_state,
            pose_link=None,
            group_name=self._group_name,
        )
        self[state_request] = trajectory

        if self._pose_link is not None:
            end_pose = pose_stamped_msg(
                pose=end_state.get_pose(self._pose_link),
                frame_id=end_state.robot_model.model_frame,
            )
            pose_request = PlanRequest(
                start_state=start_state,
                goal=end_pose,
                pose_link=self._pose_link,
                group_name=self._group_name,
            )
            self[pose_request] = trajectory

        if _reverse:
            self.cache_trajectory(
                trajectory.reverse(),
                request=request,
                validate=validate,
                _reverse=False,
            )

    def get_best_trajectory(
        self, request: PlanRequest, validate: bool = True
    ) -> RobotTrajectory:
        """Get the best (cheapest) cached trajectory for `request`."""
        trajectory = self[request][0]
        if validate:
            self._validate_trajectory_quality(trajectory, request)
        return trajectory

    def get_trajectories(
        self, request: PlanRequest, validate: bool = True
    ) -> list[RobotTrajectory]:
        """Get all cached trajectories for `request`, ranked best-first."""
        trajectories = self[request]
        if validate:
            for trajectory in trajectories:
                self._validate_trajectory_quality(trajectory, request)
        return trajectories

    def has_trajectory(self, request: PlanRequest) -> bool:
        """Check if a trajectory exists for `request`."""
        return request in self

    def delete_trajectory(self, request: PlanRequest) -> None:
        """Delete all trajectories for `request`."""
        del self[request]

    # ---------------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
