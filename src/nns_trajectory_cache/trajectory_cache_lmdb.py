"""LMDB-backed fuzzy trajectory cache.

Concrete backend that persists the cache to a single LMDB file via
memory-mapped reads. Each call into the backend runs its own
transaction; the base class serializes the read-modify-write in
`__setitem__` with `self._lock`, so concurrent threads inside a single
Python process see consistent reads.

See `trajectory_cache.FuzzyTrajectoryCache` for the abstract base class
and the geometric/fuzzy-matching pieces.
"""

import os
import pickle
from typing import Any, Literal, Optional

import lmdb
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
_DEFAULT_MAP_SIZE: int = 2 * 1024**3  # 2 GiB of virtual address space


class LMDBFuzzyTrajectoryCache(FuzzyTrajectoryCache):
    """Persistent fuzzy trajectory cache backed by a single LMDB file.

    Values are pickled and stored under their fuzzy-key bytes. Each
    backend primitive opens its own LMDB transaction; the base class's
    `_lock` guards the read-modify-write in `__setitem__`.

    On metadata mismatch (e.g. scene_hash or tolerances changed across
    runs), the LMDB file is removed from disk and recreated in place.
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
        map_size: int = _DEFAULT_MAP_SIZE,
        parent_logger: Optional[RcutilsLogger] = None,
    ):
        """
        Args:
            path: Absolute path to the cache file. The parent directory
                is created if it does not exist.
            map_size: Maximum virtual address space, in bytes, reserved
                for the LMDB environment. Cheap on Linux (it's only
                virtual); pick a value larger than the cache will ever
                grow.
            (Other args: see `FuzzyTrajectoryCache`.)
        """
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

        self._map_size = int(map_size)

        self._env: lmdb.Environment | None = None

        self._open_and_validate()

    def _require_env(self) -> lmdb.Environment:
        env = self._env
        if env is None:
            raise RuntimeError("Trajectory cache database is not open")
        return env

    # ---------------------------------------------------------------
    # Backend primitives
    # ---------------------------------------------------------------

    def _get_raw(self, key: bytes) -> Any:
        env = self._require_env()
        with env.begin(buffers=True) as txn:
            raw = txn.get(key)
            if raw is None:
                raise KeyError(key)
            return pickle.loads(raw)

    def _put_raw(self, key: bytes, value: Any) -> None:
        env = self._require_env()
        data = pickle.dumps(value, protocol=_PICKLE_PROTOCOL)
        with env.begin(write=True) as txn:
            txn.put(key, data)

    def _delete_raw(self, key: bytes) -> None:
        env = self._require_env()
        with env.begin(write=True) as txn:
            if not txn.delete(key):
                raise KeyError(key)

    def _contains_raw(self, key: bytes) -> bool:
        env = self._require_env()
        with env.begin(buffers=True) as txn:
            return txn.get(key) is not None

    def _clear_storage(self) -> None:
        """Wipe the LMDB file by closing, deleting on disk, and reopening."""
        self.close()
        self._delete_db_files()
        self.open()

    def _delete_db_files(self) -> None:
        """Remove the LMDB data file and its sibling lock file."""
        for filepath in (self._path, self._path + "-lock"):
            if os.path.exists(filepath):
                os.remove(filepath)

    def __len__(self) -> int:
        return self._require_env().stat()["entries"]

    def open(self):
        """Open the database, creating the file if it does not exist."""
        with self._lock:
            if not self._closed:
                self.log("Database is already open", severity="WARN")
                return

            self._env = lmdb.open(
                self._path,
                map_size=self._map_size,
                subdir=False,
                readahead=False,
                writemap=True,
                metasync=True,
                sync=True,
                max_readers=126,
                max_dbs=0,
                create=True,
            )
            self._closed = False

    def close(self):
        """Close the database."""
        with self._lock:
            if self._closed:
                self.log("Database is already closed", severity="WARN")
                return

            try:
                if self._env is not None:
                    self._env.close()
            finally:
                self._env = None
                self._closed = True
