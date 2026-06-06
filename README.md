# nns-trajectory-cache

A self-contained, library-agnostic **nearest-neighbour cache for
motion-planning trajectories**. It maps a planning request (a start
configuration + a goal) to one or more previously-planned trajectories
whose stored request lies within configured per-coordinate tolerances of
the query — i.e. approximate-nearest-neighbour (ANN) lookup over the
request space, so a robot can reuse a motion it has already planned
instead of re-running the planner.

This repo is a reference extraction of the cache I built and benchmarked
for a class project and then used in private robotics work. The original
lived inside a ROS 2 / MoveIt stack; **this version has no robotics
dependencies at all** — you convert whatever request / trajectory
representation you have into a few plain dataclasses and back.

## Design

The cache is a `Mapping`-like structure:

- **Key** — `TrajectoryCacheKey`: a start joint configuration plus a
  goal, where the goal is either a `JointGoal` (target joint
  configuration) or a `CartesianGoal` (target end-effector pose).
- **Value** — `TrajectoryCacheValue`: an opaque, picklable `trajectory`
  payload (whatever you want to store — a serialized message, an array,
  a list of waypoints) plus a scalar `path_cost`. Competing trajectories
  for a match are ranked cheapest-`path_cost`-first.

The cache never inspects the trajectory payload, so it doesn't care
whether it came from MoveIt, OMPL, or a hand-rolled planner. The abstract
base `TrajectoryCache` owns key validation, tolerance handling, and the
high-level helpers (`put` / `get` / `get_best` / `has` / `delete`);
concrete backends only implement how the request space is *indexed*.

### Backends

All four expose the identical API and differ only in their index
structure. `N` = entries in the cache, `B` = candidates per match group
(≤ `max_trajectories`), `d` = feature dimensionality.

| Backend | Class | Strategy | Query | Update | Persistence |
| ------- | ----- | -------- | ----- | ------ | ----------- |
| **Fuzzy bin (LMDB)** | `LMDBFuzzyTrajectoryCache` | Quantize every coordinate via `int(x // tol)` into an integer-bin key; JSON-encode it as an LMDB key. | `O(1)` hash + `O(B)` | `O(log B)` + 1 LMDB txn | LMDB file |
| **Fuzzy bin (dict)** | `DictFuzzyTrajectoryCache` | Same binning over an in-memory dict. | `O(1)` + `O(B)` | `O(log B)` | optional pickle |
| **Brute force** | `LinearTrajectoryCache` | Store exact fingerprints; linear scan with a per-coordinate tolerance check. Exact baseline. | `O(N·d)` | `O(log B)` | optional pickle |
| **k-d tree** | `KDTreeTrajectoryCache` | scipy `KDTree`s over scaled feature vectors; L∞ ball query of radius 1. | `O(log N)` amortized (+ lazy rebuild) | `O(1)` append | optional pickle |

The three approximate backends (both fuzzy variants and the k-d tree)
all reproduce the *same* per-coordinate tolerance equivalence the brute-
force baseline computes exactly, so they're directly comparable on
identical input. The fuzzy backends have one documented caveat —
**bin-boundary aliasing**: coordinates near `k · tolerance` can toggle
into an adjacent bin under tiny floating-point drift, so two physically
near requests occasionally miss each other.

## Usage

```python
from nns_trajectory_cache import (
    KDTreeTrajectoryCache, TrajectoryCacheKey, JointGoal, CartesianGoal,
)

joints = ["j0", "j1", "j2", "j3", "j4", "j5"]

cache = KDTreeTrajectoryCache(
    path=None,                       # or an absolute path to persist
    joint_names=joints,
    scene_hash="my-scene-v1",
    planning_frame="world",
    group_name="arm",
    pose_link="tool0",               # None to reject Cartesian goals
    robot_state_tolerance=0.05,      # rad; float or per-joint dict
    position_tolerance=0.01,         # m; float or (x, y, z)
    orientation_tolerance=0.05,      # float or per-quaternion-component
    max_trajectories=1,
)

with cache:  # open()/close() (loads/saves persistence if path is set)
    key = TrajectoryCacheKey(
        start_joint_positions={j: 0.0 for j in joints},
        goal=JointGoal(joint_positions={j: 0.5 for j in joints}),
        group_name="arm",
    )
    # store your trajectory (any picklable object) + its cost
    cache.put(key, my_trajectory, path_cost=2.3)

    # later — a nearby request hits the cache
    if cache.has(query_key):
        traj = cache.get_best_trajectory(query_key)
```

Swap in `LMDBFuzzyTrajectoryCache`, `DictFuzzyTrajectoryCache`, or
`LinearTrajectoryCache` by changing only the constructor — the LMDB
backend requires a `path`; the k-d tree additionally needs `joint_names`.

## Benchmark

`cache_benchmark_direct.py` is a self-contained benchmark that fills each
backend with synthetic random goals (no planner in the loop) and records
per-leg query / update time, hit-vs-miss, and cache size to a CSV —
useful for watching how each index scales with cache size.

```bash
uv run nns-cache-benchmark --n-unique-goals 5000 --output-csv bench.csv
# or:
uv run python -m nns_trajectory_cache.cache_benchmark_direct --help
```

## Install / develop

```bash
uv sync            # installs lmdb, numpy, scipy
uv run python -c "import nns_trajectory_cache"
```

Only the k-d tree backend needs numpy + scipy and only the LMDB backend
needs lmdb; the dict and linear backends are pure standard library.

## `deprecated/`

The original MoveIt-coupled pieces — the pydantic `PlanRequest` models
and the *planning* benchmark that drove a live robot — are preserved
under `deprecated/` for reference. They will not run here. See
[`deprecated/README.md`](deprecated/README.md). The
`cache_benchmark_analysis.ipynb` notebook analyzes output from that
(deprecated) planning benchmark.
