"""Nearest-neighbour trajectory cache.

A self-contained, library-agnostic cache for motion-planning
trajectories. It maps a planning request (start configuration + goal)
to one or more previously-planned trajectories whose stored request
lies within configured per-coordinate tolerances of the query —
approximate-nearest-neighbour lookup over the request space.

Callers convert whatever request / trajectory representation they have
(ROS, MoveIt, custom) into the plain dataclasses in
`nns_trajectory_cache.types` (`TrajectoryCacheKey`,
`TrajectoryCacheValue`, `JointGoal`, `CartesianGoal`) and back; the
cache itself has no robotics dependencies.

Four interchangeable backends implement the same Mapping-like API,
differing only in how they index the request space:

- `LMDBFuzzyTrajectoryCache` — integer-bin (fuzzy) keys, persisted to
  an LMDB file.
- `DictFuzzyTrajectoryCache` — the same fuzzy binning over an in-memory
  dict (optionally pickled to disk).
- `LinearTrajectoryCache` — brute-force linear scan with a
  per-coordinate tolerance check (exact baseline).
- `KDTreeTrajectoryCache` — scipy k-d trees over feature vectors with an
  L∞ ball query.
"""

from nns_trajectory_cache.trajectory_cache import (
    OrientationToleranceT,
    PositionToleranceT,
    RobotStateToleranceT,
    TrajectoryCache,
)
from nns_trajectory_cache.trajectory_cache_dict import (
    DictFuzzyTrajectoryCache,
)
from nns_trajectory_cache.trajectory_cache_fuzzy import FuzzyTrajectoryCache
from nns_trajectory_cache.trajectory_cache_kdtree import KDTreeTrajectoryCache
from nns_trajectory_cache.trajectory_cache_linear import LinearTrajectoryCache
from nns_trajectory_cache.trajectory_cache_lmdb import (
    LMDBFuzzyTrajectoryCache,
)
from nns_trajectory_cache.types import (
    CartesianGoal,
    Goal,
    JointGoal,
    TrajectoryCacheKey,
    TrajectoryCacheValue,
)

__all__ = [
    "TrajectoryCache",
    "FuzzyTrajectoryCache",
    "LMDBFuzzyTrajectoryCache",
    "DictFuzzyTrajectoryCache",
    "LinearTrajectoryCache",
    "KDTreeTrajectoryCache",
    "TrajectoryCacheKey",
    "TrajectoryCacheValue",
    "Goal",
    "JointGoal",
    "CartesianGoal",
    "RobotStateToleranceT",
    "PositionToleranceT",
    "OrientationToleranceT",
]
