"""Dict-backed fuzzy trajectory cache with optional pickle persistence.

A minimal concrete backend that stores entries in a plain Python dict.
Useful as a baseline for benchmarking and as a reference implementation
of the `FuzzyTrajectoryCache` ABC.

When constructed with a `path`, the dict is pickled to disk on `close()`
and reloaded from disk on `open()`. Because fuzzy metadata
(`b"scene_hash"`, tolerances, etc.) is stored as keys *inside* the
dict, the inherited `_validate_db()` automatically catches cross-run
config mismatches after a load and wipes the store. Constructed with
`path=None`, the cache is process-local.

`_get_raw` returns a shallow list-copy for list values so the base
class's read-modify-write in `__setitem__` operates on a private copy
(matching the pickle round-trip semantics of the LMDB backend).

See `trajectory_cache_fuzzy.FuzzyTrajectoryCache` for the abstract
base class and the geometric/fuzzy-matching pieces.
"""

import os
import pickle
from typing import Any, Literal, Optional

from rclpy.impl.rcutils_logger import RcutilsLogger

from tabletop_rig.interfaces.moveit.trajectory_cache import (
    OrientationToleranceT,
    PositionToleranceT,
    RobotStateToleranceT,
)
from tabletop_rig.interfaces.moveit.trajectory_cache_fuzzy import (
    FuzzyTrajectoryCache,
)

_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


class DictFuzzyTrajectoryCache(FuzzyTrajectoryCache):
    """Fuzzy trajectory cache backed by an in-memory Python dict.

    Args:
        path: Absolute path to a pickle file. If provided, the cache
            is loaded from this file on `open()` and saved to it on
            `close()`. If `None`, the cache is purely process-local.
        (Other args: see `FuzzyTrajectoryCache`.)
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

        self._store: dict[bytes, Any] = {}

        self._open_and_validate()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Trajectory cache is not open")

    # ---------------------------------------------------------------
    # Backend primitives
    # ---------------------------------------------------------------

    def _get_raw(self, key: bytes) -> Any:
        self._require_open()
        value = self._store[key]
        if isinstance(value, list):
            return list(value)
        return value

    def _put_raw(self, key: bytes, value: Any) -> None:
        self._require_open()
        self._store[key] = value

    def _delete_raw(self, key: bytes) -> None:
        self._require_open()
        del self._store[key]

    def _contains_raw(self, key: bytes) -> bool:
        self._require_open()
        return key in self._store

    def _clear_storage(self) -> None:
        self._require_open()
        self._store.clear()

    def __len__(self) -> int:
        self._require_open()
        return len(self._store)

    def open(self) -> None:
        """Mark the cache as open and, if `path` is set, load from disk.

        A missing or unreadable file is treated as "start fresh" — the
        in-memory store stays empty and execution continues. Metadata
        mismatch detection runs after this (inside
        `_open_and_validate` at construction).
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
