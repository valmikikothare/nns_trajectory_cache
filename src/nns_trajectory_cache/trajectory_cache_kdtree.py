"""K-d tree trajectory cache (in-memory, feature-vector indexed).

Each `TrajectoryCacheKey` is reduced to a feature vector whose
dimensionality depends on the goal type (`n` = number of joints in the
group):

- **Joint-space goal (2n-D)**: start joint angles ++ goal joint angles.
- **Cartesian goal ((n+7)-D)**: start joint angles ++ 3 goal position
  components ++ 4 goal orientation quaternion components.

Because these two spaces have different dimensions they cannot share
a single tree, so this backend holds two scipy `KDTree`s side-by-side
and dispatches on goal type at every insert and query.

Per-coordinate tolerances are folded into a fixed scale vector at
init time. Feature vectors are divided by that scale before insertion
into the tree; queries scale the query point the same way and use
`KDTree.query_ball_point(..., r=1.0, p=np.inf)`, which finds every
stored point inside the L∞ hypercube of half-side 1.0 around the
query — exactly the per-coordinate tolerance equivalence the fuzzy
and linear backends use.

`group_name`, `pose_link`, and Cartesian `frame_id` are dropped from
the feature vector entirely (they live in the base class as cache-
level metadata and are validated on every request).
"""

import logging
import os
import pickle
from typing import Literal, Optional

import numpy as np
from scipy.spatial import KDTree

from nns_trajectory_cache.trajectory_cache import (
    OrientationToleranceT,
    PositionToleranceT,
    RobotStateToleranceT,
    TrajectoryCache,
)
from nns_trajectory_cache.types import (
    CartesianGoal,
    JointGoal,
    TrajectoryCacheKey,
    TrajectoryCacheValue,
)

_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


class KDTreeTrajectoryCache(TrajectoryCache):
    """In-memory trajectory cache indexed by k-d trees over feature vectors.

    Each insert appends a single (feature, value) pair to the
    appropriate store (joint-space or Cartesian); queries do an L∞
    ball search on the scaled feature space and return the cheapest
    `max_trajectories` matches.

    Tree builds are lazy and size-invalidated: a burst of inserts
    followed by one query pays exactly one tree build, but
    interleaved insert/query workloads rebuild on every query. This
    is the natural cadence for a static k-d tree backend used as a
    benchmark target.

    Args:
        path: Absolute path to a pickle file. If provided, the four
            feature/value lists are loaded from this file on `open()`
            and saved on `close()`. If `None`, the cache is purely
            process-local.
        joint_names: The canonical joint ordering for the group. Every
            `TrajectoryCacheKey` passed to the cache must supply exactly
            these joint names (in its `start_joint_positions` and any
            `JointGoal`). The ordering fixes how feature vectors are
            laid out, so it must be stable across the cache's lifetime.
        (See `TrajectoryCache`. `max_trajectories` caps the number of
        results returned per query — there is no per-point insert
        cap; each insert is its own tree point.)
    """

    def __init__(
        self,
        *,
        path: Optional[str] = None,
        scene_hash: str,
        planning_frame: str,
        group_name: str,
        pose_link: Optional[str] = None,
        joint_names: list[str],
        robot_state_tolerance: RobotStateToleranceT,
        position_tolerance: PositionToleranceT,
        orientation_tolerance: OrientationToleranceT,
        sort_by: Literal["path_length", "path_duration"] = "path_duration",
        max_trajectories: int = 1,
        logger: Optional[logging.Logger] = None,
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
            logger=logger,
        )

        # The canonical joint ordering fixes the feature-vector layout
        # and the per-coordinate scale vectors, both built once up
        # front. Every key handed to the cache must use exactly these
        # joints.
        if not joint_names:
            raise ValueError("'joint_names' must be non-empty")
        self._joint_names: list[str] = list(joint_names)
        self._state_scale: np.ndarray = self._build_state_scale()
        self._pose_scale: np.ndarray = self._build_pose_scale()

        # Joint-space goal store: 12D features.
        self._state_features: list[np.ndarray] = []
        self._state_values: list[TrajectoryCacheValue] = []
        self._state_tree: Optional[KDTree] = None
        self._state_tree_size: int = 0

        # Cartesian goal store: 13D features.
        self._pose_features: list[np.ndarray] = []
        self._pose_values: list[TrajectoryCacheValue] = []
        self._pose_tree: Optional[KDTree] = None
        self._pose_tree_size: int = 0

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Trajectory cache is not open")

    # ---------------------------------------------------------------
    # Scale vectors (built once in __init__)
    # ---------------------------------------------------------------

    def _joint_tol_vector(self) -> np.ndarray:
        """Return the (n_joints,) per-joint tolerance vector."""
        tol = self._robot_state_tolerance
        if isinstance(tol, dict):
            return np.array([tol[j] for j in self._joint_names], dtype=float)
        return np.full(len(self._joint_names), float(tol))

    def _build_state_scale(self) -> np.ndarray:
        """Build the 12D scale vector for joint-space goals."""
        joint_tol = self._joint_tol_vector()
        return np.concatenate([joint_tol, joint_tol])

    def _build_pose_scale(self) -> np.ndarray:
        """Build the 13D scale vector for Cartesian goals."""
        joint_tol = self._joint_tol_vector()
        pos_tol = self._position_tolerance
        if isinstance(pos_tol, (int, float)):
            pos_vec = np.full(3, float(pos_tol))
        else:
            pos_vec = np.array(pos_tol, dtype=float)
        ori_tol = self._orientation_tolerance
        if isinstance(ori_tol, (int, float)):
            ori_vec = np.full(4, float(ori_tol))
        else:
            ori_vec = np.array(ori_tol, dtype=float)
        return np.concatenate([joint_tol, pos_vec, ori_vec])

    # ---------------------------------------------------------------
    # Feature construction
    # ---------------------------------------------------------------

    def _joints_to_array(
        self, positions: "dict[str, float]"
    ) -> np.ndarray:
        """Pull joint positions into a numpy array in canonical order."""
        return np.array([positions[j] for j in self._joint_names], dtype=float)

    def _state_feature(self, key: TrajectoryCacheKey) -> np.ndarray:
        """Compute the 12D feature vector for a joint-space goal key."""
        assert isinstance(key.goal, JointGoal)
        start = self._joints_to_array(dict(key.start_joint_positions))
        goal = self._joints_to_array(dict(key.goal.joint_positions))
        return np.concatenate([start, goal])

    def _pose_feature(self, key: TrajectoryCacheKey) -> np.ndarray:
        """Compute the 13D feature vector for a Cartesian goal key."""
        assert isinstance(key.goal, CartesianGoal)
        start = self._joints_to_array(dict(key.start_joint_positions))
        pos = np.asarray(key.goal.position, dtype=float)
        ori = np.asarray(key.goal.orientation, dtype=float)
        return np.concatenate([start, pos, ori])

    # ---------------------------------------------------------------
    # Lazy tree build
    # ---------------------------------------------------------------

    def _ensure_state_tree(self) -> None:
        if not self._state_features:
            self._state_tree = None
            self._state_tree_size = 0
            return
        if self._state_tree is None or self._state_tree_size != len(
            self._state_features
        ):
            scaled = np.stack(self._state_features) / self._state_scale
            self._state_tree = KDTree(scaled)
            self._state_tree_size = len(self._state_features)

    def _ensure_pose_tree(self) -> None:
        if not self._pose_features:
            self._pose_tree = None
            self._pose_tree_size = 0
            return
        if self._pose_tree is None or self._pose_tree_size != len(
            self._pose_features
        ):
            scaled = np.stack(self._pose_features) / self._pose_scale
            self._pose_tree = KDTree(scaled)
            self._pose_tree_size = len(self._pose_features)

    # ---------------------------------------------------------------
    # Mapping API
    # ---------------------------------------------------------------

    def __setitem__(
        self, key: TrajectoryCacheKey, value: TrajectoryCacheValue
    ) -> None:
        """Append a single (feature, value) point to the appropriate store.

        The tree is invalidated implicitly via the size-check in
        `_ensure_*_tree` — the next query rebuilds it.
        """
        self._validate_key(key)
        if not isinstance(value, TrajectoryCacheValue):
            raise TypeError(
                f"value must be a TrajectoryCacheValue: "
                f"{type(value).__name__}"
            )
        self._require_open()

        with self._lock:
            if isinstance(key.goal, JointGoal):
                feature = self._state_feature(key)
                self._state_features.append(feature)
                self._state_values.append(value)
            else:
                feature = self._pose_feature(key)
                self._pose_features.append(feature)
                self._pose_values.append(value)

    def __getitem__(
        self, key: TrajectoryCacheKey
    ) -> list[TrajectoryCacheValue]:
        """Return cheapest-`max_trajectories` matches via L∞ ball query."""
        self._validate_key(key)
        self._require_open()

        if isinstance(key.goal, JointGoal):
            self._ensure_state_tree()
            tree = self._state_tree
            values = self._state_values
            query_feature = self._state_feature(key)
            scale = self._state_scale
        else:
            self._ensure_pose_tree()
            tree = self._pose_tree
            values = self._pose_values
            query_feature = self._pose_feature(key)
            scale = self._pose_scale

        if tree is None:
            raise KeyError(key)

        scaled_query = query_feature / scale
        indices = tree.query_ball_point(scaled_query, r=1.0, p=np.inf)

        if not indices:
            raise KeyError(key)

        candidates = sorted(values[i] for i in indices)
        return candidates[: self._max_trajectories]

    def __contains__(self, key: TrajectoryCacheKey) -> bool:
        """Return True iff at least one stored point is within tolerance."""
        self._validate_key(key)
        self._require_open()

        if isinstance(key.goal, JointGoal):
            self._ensure_state_tree()
            tree = self._state_tree
            query_feature = self._state_feature(key)
            scale = self._state_scale
        else:
            self._ensure_pose_tree()
            tree = self._pose_tree
            query_feature = self._pose_feature(key)
            scale = self._pose_scale

        if tree is None:
            return False
        scaled_query = query_feature / scale
        indices = tree.query_ball_point(scaled_query, r=1.0, p=np.inf)
        return bool(indices)

    def __delitem__(self, key: TrajectoryCacheKey) -> None:
        """Delete every stored point within tolerance of `key`.

        Bypasses the k-d tree (which has no native delete) and rebuilds
        the affected store by filtering. Invalidates the tree.

        Raises:
            KeyError: If no stored point matches.
        """
        self._validate_key(key)
        self._require_open()

        is_joint_goal = isinstance(key.goal, JointGoal)
        with self._lock:
            if is_joint_goal:
                features = self._state_features
                values = self._state_values
                scale = self._state_scale
                query_feature = self._state_feature(key)
            else:
                features = self._pose_features
                values = self._pose_values
                scale = self._pose_scale
                query_feature = self._pose_feature(key)

            keep_features: list[np.ndarray] = []
            keep_values: list[TrajectoryCacheValue] = []
            matched = False
            for feat, val in zip(features, values):
                if np.all(np.abs(feat - query_feature) <= scale):
                    matched = True
                    continue
                keep_features.append(feat)
                keep_values.append(val)

            if not matched:
                raise KeyError(key)

            if is_joint_goal:
                self._state_features = keep_features
                self._state_values = keep_values
                self._state_tree = None
                self._state_tree_size = 0
            else:
                self._pose_features = keep_features
                self._pose_values = keep_values
                self._pose_tree = None
                self._pose_tree_size = 0

    def __len__(self) -> int:
        """Total number of stored points across both trees."""
        self._require_open()
        return len(self._state_features) + len(self._pose_features)

    def open(self) -> None:
        """Mark the cache as open and, if `path` is set, load from disk.

        The persisted payload is `(joint_names, state_features,
        state_values, pose_features, pose_values)`. If the persisted
        `joint_names` does not match the configured ordering, the loaded
        features would be scrambled against the cache's scale vectors —
        so we discard the file and start fresh in that case.

        Trees are not persisted; they're rebuilt lazily on the next
        query.
        """
        with self._lock:
            if not self._closed:
                self.log("Cache is already open", severity="WARN")
                return
            if self._path is not None and os.path.exists(self._path):
                try:
                    with open(self._path, "rb") as f:
                        payload = pickle.load(f)
                    (
                        saved_joint_names,
                        state_features,
                        state_values,
                        pose_features,
                        pose_values,
                    ) = payload
                    if list(saved_joint_names) != self._joint_names:
                        self.log(
                            f"Joint ordering in {self._path} does not match "
                            f"the configured joint_names; starting fresh.",
                            severity="WARN",
                        )
                    else:
                        self._state_features = list(state_features)
                        self._state_values = list(state_values)
                        self._pose_features = list(pose_features)
                        self._pose_values = list(pose_values)
                        self._state_tree = None
                        self._state_tree_size = 0
                        self._pose_tree = None
                        self._pose_tree_size = 0
                        self.log(
                            f"Loaded {len(self._state_features)} state + "
                            f"{len(self._pose_features)} pose entries from "
                            f"{self._path}",
                            severity="INFO",
                        )
                except Exception as e:
                    self.log(
                        f"Failed to load cache from {self._path}: {e}. "
                        f"Starting fresh.",
                        severity="WARN",
                    )
            self._closed = False

    def close(self) -> None:
        """Mark the cache as closed and, if `path` is set, save to disk."""
        with self._lock:
            if self._closed:
                self.log("Cache is already closed", severity="WARN")
                return
            if self._path is not None:
                try:
                    payload = (
                        list(self._joint_names),
                        self._state_features,
                        self._state_values,
                        self._pose_features,
                        self._pose_values,
                    )
                    with open(self._path, "wb") as f:
                        pickle.dump(payload, f, protocol=_PICKLE_PROTOCOL)
                    self.log(
                        f"Saved {len(self._state_features)} state + "
                        f"{len(self._pose_features)} pose entries to "
                        f"{self._path}",
                        severity="INFO",
                    )
                except Exception as e:
                    self.log(
                        f"Failed to save cache to {self._path}: {e}",
                        severity="ERROR",
                    )
            self._closed = True
