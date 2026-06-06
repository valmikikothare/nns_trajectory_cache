"""Trajectory caching for motion planning (abstract base).

This module defines the storage-agnostic abstract base class
`TrajectoryCache`, which exposes a `Mapping`-like API keyed on
`TrajectoryCacheKey` and valued on lists of `TrajectoryCacheValue`. The
base class does not know how requests are indexed — that's the job of
concrete backends — nor does it depend on ROS, MoveIt, or any robotics
library. Callers convert whatever planning request / trajectory they
have into the plain dataclasses in `nns_trajectory_cache.types` before
calling in.

Concrete backends subclass `TrajectoryCache` and implement the Mapping
primitives (`__setitem__`, `__getitem__`, `__contains__`,
`__delitem__`) plus the lifecycle hooks (`open`, `close`, `__len__`).

A fuzzy-binning intermediate class (`FuzzyTrajectoryCache`) lives in
`trajectory_cache_fuzzy`; the linear, k-d tree, dict, and LMDB backends
build on top of these.

Classes:
    TrajectoryCache: Abstract base class for trajectory caches keyed on
        TrajectoryCacheKey.
"""

import abc
import logging
import os
import threading
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional

from nns_trajectory_cache.types import (
    CartesianGoal,
    JointGoal,
    TrajectoryCacheKey,
    TrajectoryCacheValue,
)

RobotStateToleranceT = float | dict[str, float]
PositionToleranceT = float | tuple[float, float, float]
OrientationToleranceT = (
    float | tuple[float, float, float] | tuple[float, float, float, float]
)

_SEVERITY_TO_LEVEL: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "FATAL": logging.CRITICAL,
}


def _is_iterable(value: Any) -> bool:
    """Return True if `value` is a non-string iterable."""
    return isinstance(value, Iterable) and not isinstance(
        value, (str, bytes)
    )


class TrajectoryCache(metaclass=abc.ABCMeta):
    """Abstract base for trajectory caches keyed on `TrajectoryCacheKey`.

    Subclasses provide the Mapping primitives (`__setitem__`,
    `__getitem__`, `__contains__`, `__delitem__`) and the lifecycle
    hooks (`open`, `close`, `__len__`). Everything else — high-level
    caching helpers, request validation, tolerance management — lives
    here.

    The Mapping API treats `TrajectoryCacheKey` as the key (start config
    + goal + group name + optional pose link) and a list of
    `TrajectoryCacheValue` as the value. `__getitem__` returns those
    values ranked best-first (cheapest `path_cost` first); subclasses
    are responsible for honoring that ranking.

    Args:
        path: Absolute path to a persistence file, or `None` for a
            purely in-memory cache. `~` and `$VAR`s are expanded.
        scene_hash: Hash describing the static scene/rig configuration.
            Subclasses may use it to detect stale persistent state and
            wipe it.
        planning_frame: The frame all Cartesian goals must live in.
        group_name: The joint group every request must address. Stored
            once as cache metadata rather than per-key, since it never
            varies for a given cache instance.
        pose_link: The end-effector link Cartesian goals are expressed
            against. Like `group_name`, stored once as metadata. May be
            `None`, in which case the cache only accepts joint-space
            goals.
        robot_state_tolerance: Per-joint angle tolerance. Used by fuzzy
            backends for binning and by the linear / k-d tree backends
            for matching.
        position_tolerance: Cartesian goal position tolerance.
        orientation_tolerance: Cartesian goal orientation tolerance.
        sort_by: Label describing what `path_cost` on cached values
            means (`path_length` or `path_duration`). Stored as
            metadata; the cache always ranks by `path_cost` ascending.
        max_trajectories: Cap on cached trajectories kept per match
            group. Subclasses define what a "match group" means.
        logger: Optional `logging.Logger`. Defaults to a module logger.
    """

    def __init__(
        self,
        *,
        path: Optional[str] = None,
        scene_hash: str,
        planning_frame: str,
        group_name: str,
        pose_link: Optional[str] = None,
        robot_state_tolerance: RobotStateToleranceT,
        position_tolerance: PositionToleranceT,
        orientation_tolerance: OrientationToleranceT,
        sort_by: Literal["path_length", "path_duration"] = "path_duration",
        max_trajectories: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        self._logger = logger or logging.getLogger(
            "nns_trajectory_cache"
        )

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

    # ---------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------

    def get_logger(self) -> logging.Logger:
        """Get the logger instance."""
        return self._logger

    def log(self, msg: str, severity: str = "INFO") -> None:
        """Log `msg` at the given severity (ROS-style severity names)."""
        self._logger.log(_SEVERITY_TO_LEVEL.get(severity, logging.INFO), msg)

    @property
    def path(self) -> Optional[str]:
        """The path to the persistence file, or None if in-memory."""
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
            if not robot_state_tolerance:
                raise ValueError("robot_state_tolerance must be non-empty")
            if any(x <= 0 for x in robot_state_tolerance.values()):
                raise ValueError("robot_state_tolerance must be positive")
        elif robot_state_tolerance <= 0:
            raise ValueError("robot_state_tolerance must be positive")

        if _is_iterable(position_tolerance):
            position_tolerance = tuple(map(float, position_tolerance))
            if len(position_tolerance) != 3:
                raise ValueError("position_tolerance must be a 3-tuple")
            if any(x <= 0 for x in position_tolerance):
                raise ValueError("position_tolerance must be positive")
        elif position_tolerance <= 0:
            raise ValueError("position_tolerance must be positive")

        if _is_iterable(orientation_tolerance):
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
    def _normalize_path(path: Optional[str]) -> Optional[str]:
        """Normalize and validate a persistence file path.

        Returns the absolute path with `~` and `$VAR`s expanded, after
        ensuring the parent directory exists. Returns `None` unchanged
        (a purely in-memory cache).

        Raises:
            ValueError: If `path` is relative or names something that
                already exists and is not a regular file.
        """
        if path is None:
            return None
        path = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isabs(path):
            raise ValueError(f"Trajectory cache path must be absolute: {path}")
        if os.path.exists(path) and not os.path.isfile(path):
            raise ValueError(
                f"Trajectory cache path must be a regular file: {path}"
            )
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
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
        """The joint group this cache is configured for."""
        return self._group_name

    @property
    def pose_link(self) -> Optional[str]:
        """The end-effector link Cartesian goals are expressed against.

        `None` means this cache rejects Cartesian goals entirely.
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
        """Label describing what cached `path_cost` values mean."""
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
        self, key: TrajectoryCacheKey, value: TrajectoryCacheValue
    ) -> None:
        """Insert a trajectory value under the given key.

        Subclasses define what "insert" means — appending to a ranked
        list, eviction policy, etc.
        """

    @abc.abstractmethod
    def __getitem__(
        self, key: TrajectoryCacheKey
    ) -> list[TrajectoryCacheValue]:
        """Return matching values for the key, ranked best-first.

        Raises:
            KeyError: If no matching entry is found.
        """

    @abc.abstractmethod
    def __contains__(self, key: TrajectoryCacheKey) -> bool:
        """Return True iff at least one matching value exists."""

    @abc.abstractmethod
    def __delitem__(self, key: TrajectoryCacheKey) -> None:
        """Delete all matching values for the key.

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
    # Key validation
    # ---------------------------------------------------------------

    def _validate_key(self, key: TrajectoryCacheKey) -> None:
        """Validate that `key` is well-formed for cache I/O.

        Subclasses may rely on these invariants when implementing the
        Mapping API.
        """
        if not isinstance(key, TrajectoryCacheKey):
            raise TypeError(
                f"key must be a TrajectoryCacheKey: {type(key).__name__}"
            )

        start = key.start_joint_positions
        if not isinstance(start, Mapping) or not start:
            raise TypeError(
                "key.start_joint_positions must be a non-empty mapping of "
                f"joint name -> position: {start!r}"
            )

        if key.group_name != self._group_name:
            raise ValueError(
                f"Key group_name {key.group_name!r} does not match the "
                f"cache's configured group_name {self._group_name!r}"
            )

        goal = key.goal
        if isinstance(goal, JointGoal):
            if not goal.joint_positions:
                raise ValueError("JointGoal.joint_positions must be non-empty")
            if key.pose_link is not None:
                raise ValueError(
                    f"pose_link must be None for a joint-space goal: "
                    f"{key.pose_link!r}"
                )
        elif isinstance(goal, CartesianGoal):
            if self._pose_link is None:
                raise ValueError(
                    "Cache was configured without a 'pose_link'; Cartesian "
                    "goals are not accepted"
                )
            if goal.frame_id != self._planning_frame:
                raise ValueError(
                    f"Goal pose frame_id must be '{self._planning_frame}': "
                    f"{goal.frame_id}"
                )
            if key.pose_link is None:
                raise ValueError(
                    "pose_link must be provided for a Cartesian goal"
                )
            if key.pose_link != self._pose_link:
                raise ValueError(
                    f"Key pose_link {key.pose_link!r} does not match the "
                    f"cache's configured pose_link {self._pose_link!r}"
                )
        else:
            raise TypeError(
                f"key.goal must be a JointGoal or CartesianGoal: "
                f"{type(goal).__name__}"
            )

    # ---------------------------------------------------------------
    # High-level API (concrete, in terms of Mapping)
    # ---------------------------------------------------------------

    def put(
        self,
        key: TrajectoryCacheKey,
        trajectory: Any,
        *,
        path_cost: float,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Cache `trajectory` under `key`, ranked by `path_cost`.

        Convenience wrapper around `self[key] = TrajectoryCacheValue(...)`.
        """
        self[key] = TrajectoryCacheValue(
            trajectory=trajectory,
            path_cost=path_cost,
            metadata=dict(metadata or {}),
        )

    def get(self, key: TrajectoryCacheKey) -> list[TrajectoryCacheValue]:
        """Get all cached values for `key`, ranked best-first.

        Raises:
            KeyError: If no matching entry is found.
        """
        return self[key]

    def get_best(self, key: TrajectoryCacheKey) -> TrajectoryCacheValue:
        """Get the single best (cheapest) cached value for `key`.

        Raises:
            KeyError: If no matching entry is found.
        """
        return self[key][0]

    def get_trajectories(self, key: TrajectoryCacheKey) -> list[Any]:
        """Get all cached trajectory payloads for `key`, ranked best-first."""
        return [v.trajectory for v in self[key]]

    def get_best_trajectory(self, key: TrajectoryCacheKey) -> Any:
        """Get the best (cheapest) cached trajectory payload for `key`."""
        return self[key][0].trajectory

    def has(self, key: TrajectoryCacheKey) -> bool:
        """Check if a value exists for `key`."""
        return key in self

    def delete(self, key: TrajectoryCacheKey) -> None:
        """Delete all values for `key`."""
        del self[key]

    # ---------------------------------------------------------------
    # Context manager
    # ---------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
