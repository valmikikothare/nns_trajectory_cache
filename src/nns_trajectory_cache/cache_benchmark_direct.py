"""Standalone, self-contained trajectory-cache backend benchmark.

Exercises the four cache backends (`lmdb`, `dict`, `linear`, `kdtree`)
directly, with no ROS / MoveIt / planner in the loop. Every "miss" is
satisfied by a synthetic dummy trajectory built to exactly bridge the
request's start and goal, so the cache fills orders of magnitude faster
than a real planning loop would allow — which lets us push the goal
count high and watch how query / update time scale with cache size.

For each configured backend the task:

1. Builds a fresh backend instance (LMDB persists to a temp file; the
   in-memory backends run with `path=None`).
2. Generates `n_unique_goals` random goals around a synthetic `idle`
   joint configuration. Each goal is, 50/50, a joint-space goal or a
   Cartesian goal (whose pose is produced by a deterministic synthetic
   forward-kinematics map so round-trips are reproducible).
3. For every leg (`idle -> goal` and `goal -> idle`):
   - Builds the matching `TrajectoryCacheKey`.
   - Times a cache query (`cache.get`).
   - On miss, builds a dummy trajectory payload + path cost and times
     `cache.put(...)`.

Per leg we record query/update times, hit-vs-miss, and the cache size
before and after any insert. Rows are written to a CSV.

Run it:

    python -m nns_trajectory_cache.cache_benchmark_direct \\
        --backends lmdb dict linear kdtree \\
        --n-unique-goals 5000 --output-csv /tmp/cache_benchmark_direct.csv
"""

import argparse
import csv
import math
import os
import tempfile
import time
import traceback
from typing import Any, Optional

import numpy as np

from nns_trajectory_cache.trajectory_cache import TrajectoryCache
from nns_trajectory_cache.trajectory_cache_dict import (
    DictFuzzyTrajectoryCache,
)
from nns_trajectory_cache.trajectory_cache_kdtree import KDTreeTrajectoryCache
from nns_trajectory_cache.trajectory_cache_linear import LinearTrajectoryCache
from nns_trajectory_cache.trajectory_cache_lmdb import (
    LMDBFuzzyTrajectoryCache,
)
from nns_trajectory_cache.types import (
    CartesianGoal,
    JointGoal,
    TrajectoryCacheKey,
)

_VALID_BACKENDS = ("lmdb", "dict", "linear", "kdtree")

_CSV_FIELDS = (
    "backend",
    "phase",
    "cycle",
    "goal_idx",
    "goal_type",
    "direction",
    "hit",
    "cache_size_before",
    "cache_size_after",
    "query_time_s",
    "update_time_s",
    "success",
    "error",
)

_PLANNING_FRAME = "world"
_GROUP_NAME = "arm"
_POSE_LINK = "tool0"
_SCENE_HASH = "synthetic-benchmark-scene"

# Dummy trajectory cost floor so `sort_by="path_duration"` style ranking
# always sees a positive, finite cost.
_MIN_PATH_COST = 1e-3


class DirectCacheBenchmark:
    """Benchmark trajectory-cache backends via direct query/update.

    Args:
        backends: Subset of {"lmdb", "dict", "linear", "kdtree"} to run.
        n_joints: Number of joints in the synthetic arm.
        n_unique_goals: Distinct random goals per backend (one
            round-trip each).
        n_cycles: Replay cycles over the goal sequence after the collect
            pass (these should be pure hits once the cache is warm).
        seed: PRNG seed; the same seed gives the same goal sequence
            across backends so timings are comparable.
        joint_offset_range: Each joint is sampled uniformly from
            `idle ± this` (radians).
        robot_state_tolerance / position_tolerance / orientation_tolerance:
            Cache matching tolerances.
        max_trajectories: Per-match cap.
        output_csv: Where to write the per-leg results.
        cache_dir: Directory for the LMDB backend's temp file. Defaults
            to a fresh system temp dir.
    """

    def __init__(
        self,
        *,
        backends: Optional[list[str]] = None,
        n_joints: int = 6,
        n_unique_goals: int = 5000,
        n_cycles: int = 1,
        seed: int = 42,
        joint_offset_range: float = 0.3,
        robot_state_tolerance: float = 0.05,
        position_tolerance: float = 0.01,
        orientation_tolerance: float = 0.05,
        max_trajectories: int = 1,
        output_csv: str = "cache_benchmark_direct.csv",
        cache_dir: Optional[str] = None,
    ) -> None:
        backends = list(backends) if backends else list(_VALID_BACKENDS)
        unknown = set(backends) - set(_VALID_BACKENDS)
        if unknown:
            raise ValueError(
                f"Unknown cache backends: {sorted(unknown)}. "
                f"Expected a subset of: {_VALID_BACKENDS}"
            )
        self._backends = backends

        if n_joints < 1:
            raise ValueError("'n_joints' must be at least 1")
        self._n_joints = n_joints
        self._joint_names = [f"joint_{i}" for i in range(n_joints)]
        self._idle = np.zeros(n_joints, dtype=float)

        if n_unique_goals < 1:
            raise ValueError("'n_unique_goals' must be at least 1")
        self._n_unique_goals = n_unique_goals

        if n_cycles < 0:
            raise ValueError("'n_cycles' must be non-negative")
        self._n_cycles = n_cycles

        self._seed = seed
        self._joint_offset_range = float(joint_offset_range)
        if self._joint_offset_range <= 0:
            raise ValueError("'joint_offset_range' must be positive")

        self._robot_state_tolerance = robot_state_tolerance
        self._position_tolerance = position_tolerance
        self._orientation_tolerance = orientation_tolerance
        self._max_trajectories = max_trajectories

        self._output_csv = os.path.expandvars(os.path.expanduser(output_csv))
        out_parent = os.path.dirname(self._output_csv)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)

        self._cache_dir = cache_dir or tempfile.mkdtemp(prefix="nns_cache_benchmark_")
        os.makedirs(self._cache_dir, exist_ok=True)

        self._rows: list[dict[str, Any]] = []

    # ---------------------------------------------------------------
    # Cache construction
    # ---------------------------------------------------------------

    def _build_cache(self, backend: str) -> TrajectoryCache:
        common: dict[str, Any] = dict(
            scene_hash=_SCENE_HASH,
            planning_frame=_PLANNING_FRAME,
            group_name=_GROUP_NAME,
            pose_link=_POSE_LINK,
            robot_state_tolerance=self._robot_state_tolerance,
            position_tolerance=self._position_tolerance,
            orientation_tolerance=self._orientation_tolerance,
            max_trajectories=self._max_trajectories,
        )

        match backend:
            case "lmdb":
                path = os.path.join(self._cache_dir, "benchmark_lmdb")
                for f in (path, path + "-lock"):
                    if os.path.exists(f):
                        os.remove(f)
                return LMDBFuzzyTrajectoryCache(path=path, **common)
            case "dict":
                return DictFuzzyTrajectoryCache(path=None, **common)
            case "linear":
                return LinearTrajectoryCache(path=None, **common)
            case "kdtree":
                return KDTreeTrajectoryCache(
                    path=None, joint_names=self._joint_names, **common
                )
            case _:
                raise AssertionError(f"unreachable backend: {backend!r}")

    # ---------------------------------------------------------------
    # Synthetic kinematics + goal generation
    # ---------------------------------------------------------------

    def _fake_fk(
        self, joints: np.ndarray
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Deterministic synthetic forward kinematics.

        Maps a joint vector to a (position, unit-quaternion) pose. It's
        not a real robot's FK — it only needs to be a stable, smooth
        function so that the same joint configuration always yields the
        same pose (round-trips hit) while distinct configurations spread
        out across pose space.
        """
        idx = np.arange(1, self._n_joints + 1, dtype=float)
        x = float(np.sum(np.cos(joints) * idx) * 0.1)
        y = float(np.sum(np.sin(joints) * idx) * 0.1)
        z = float(np.sum(joints) * 0.05)

        half = float(np.sum(joints)) * 0.5
        axis = np.array(
            [
                math.sin(joints[0]),
                math.cos(joints[0]),
                math.sin(joints[-1]),
            ],
            dtype=float,
        )
        norm = float(np.linalg.norm(axis))
        axis = axis / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        s = math.sin(half)
        quat = (
            float(axis[0] * s),
            float(axis[1] * s),
            float(axis[2] * s),
            float(math.cos(half)),
        )
        return (x, y, z), quat

    def _joint_dict(self, joints: np.ndarray) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self._joint_names, joints)}

    def _random_goal(self, rng: np.random.Generator) -> tuple[str, np.ndarray]:
        """Return (goal_type, goal_joint_vector).

        `goal_type` is "robot_state" or "pose"; the joint vector is the
        configuration the goal represents (for pose goals it's the
        config that produced the pose, kept so the return leg's start
        matches).
        """
        offsets = rng.uniform(
            -self._joint_offset_range,
            self._joint_offset_range,
            size=self._n_joints,
        )
        joints = self._idle + offsets
        goal_type = "pose" if rng.random() < 0.5 else "robot_state"
        return goal_type, joints

    def _gen_goals(self) -> list[tuple[str, np.ndarray]]:
        rng = np.random.default_rng(self._seed)
        return [self._random_goal(rng) for _ in range(self._n_unique_goals)]

    # ---------------------------------------------------------------
    # Key + dummy-trajectory construction
    # ---------------------------------------------------------------

    def _make_key(
        self,
        start: np.ndarray,
        goal_type: str,
        goal_joints: np.ndarray,
    ) -> TrajectoryCacheKey:
        if goal_type == "pose":
            position, orientation = self._fake_fk(goal_joints)
            goal = CartesianGoal(
                position=position,
                orientation=orientation,
                frame_id=_PLANNING_FRAME,
            )
            pose_link: Optional[str] = _POSE_LINK
        else:
            goal = JointGoal(joint_positions=self._joint_dict(goal_joints))
            pose_link = None
        return TrajectoryCacheKey(
            start_joint_positions=self._joint_dict(start),
            goal=goal,
            group_name=_GROUP_NAME,
            pose_link=pose_link,
        )

    @staticmethod
    def _dummy_trajectory(
        start: np.ndarray, end: np.ndarray
    ) -> tuple[list[list[float]], float]:
        """A 2-waypoint trajectory payload and its path cost.

        The payload is just the two endpoint joint vectors (picklable,
        which the persistent backends require). The cost is the L2
        distance between them, floored so it's strictly positive.
        """
        waypoints = [list(map(float, start)), list(map(float, end))]
        cost = max(float(np.linalg.norm(end - start)), _MIN_PATH_COST)
        return waypoints, cost

    # ---------------------------------------------------------------
    # Single leg: query, then update on miss
    # ---------------------------------------------------------------

    def _process_leg(
        self,
        *,
        cache: TrajectoryCache,
        start: np.ndarray,
        goal_type: str,
        goal_joints: np.ndarray,
        backend: str,
        phase: str,
        cycle: int,
        goal_idx: int,
        direction: str,
    ) -> None:
        key = self._make_key(start, goal_type, goal_joints)

        hit: Optional[bool] = None
        query_time = float("nan")
        update_time = float("nan")
        cache_size_before = -1
        cache_size_after: Optional[int] = None
        success = False
        error = ""

        try:
            cache_size_before = len(cache)

            t0 = time.perf_counter()
            try:
                cache.get(key)
                hit = True
            except KeyError:
                hit = False
            query_time = time.perf_counter() - t0

            if not hit:
                trajectory, cost = self._dummy_trajectory(start, goal_joints)
                t0 = time.perf_counter()
                cache.put(key, trajectory, path_cost=cost)
                update_time = time.perf_counter() - t0

            cache_size_after = len(cache)
            success = True
        except Exception as e:  # noqa: BLE001 — benchmark records, continues
            error = f"{type(e).__name__}: {e}"
            print(
                f"[{backend}] {phase} cycle={cycle} goal={goal_idx} "
                f"{direction} error: {error}\n{traceback.format_exc()}"
            )

        self._rows.append(
            {
                "backend": backend,
                "phase": phase,
                "cycle": cycle,
                "goal_idx": goal_idx,
                "goal_type": goal_type,
                "direction": direction,
                "hit": hit,
                "cache_size_before": cache_size_before,
                "cache_size_after": (
                    "" if cache_size_after is None else cache_size_after
                ),
                "query_time_s": ("" if math.isnan(query_time) else f"{query_time:.9f}"),
                "update_time_s": (
                    "" if math.isnan(update_time) else f"{update_time:.9f}"
                ),
                "success": success,
                "error": error,
            }
        )

    def _do_round_trip(
        self,
        *,
        cache: TrajectoryCache,
        goal_type: str,
        goal_joints: np.ndarray,
        backend: str,
        phase: str,
        cycle: int,
        goal_idx: int,
    ) -> None:
        """idle -> goal -> idle (cache-only; nothing is executed)."""
        # Forward leg: idle -> goal.
        self._process_leg(
            cache=cache,
            start=self._idle,
            goal_type=goal_type,
            goal_joints=goal_joints,
            backend=backend,
            phase=phase,
            cycle=cycle,
            goal_idx=goal_idx,
            direction="to_goal",
        )
        # Return leg: goal-config -> idle (always a joint-space goal).
        self._process_leg(
            cache=cache,
            start=goal_joints,
            goal_type="robot_state",
            goal_joints=self._idle,
            backend=backend,
            phase=phase,
            cycle=cycle,
            goal_idx=goal_idx,
            direction="to_idle",
        )

    # ---------------------------------------------------------------
    # Per-backend runner
    # ---------------------------------------------------------------

    def _run_backend(self, backend: str, goals: list[tuple[str, np.ndarray]]) -> None:
        print(f"=== Direct cache benchmark backend: {backend} ===")
        cache = self._build_cache(backend)
        with cache:
            # Collect phase — cache grows roughly monotonically, giving
            # clean time-vs-size data.
            for i, (goal_type, goal_joints) in enumerate(goals):
                self._do_round_trip(
                    cache=cache,
                    goal_type=goal_type,
                    goal_joints=goal_joints,
                    backend=backend,
                    phase="collect",
                    cycle=0,
                    goal_idx=i,
                )
                if i % 500 == 0:
                    print(f"[{backend}] collect progress: {i}/{len(goals)}")

            print(f"[{backend}] collect done, cache_size={len(cache)}")

            # Cycle phase — replays. Every leg should now be a hit
            # (modulo fuzzy bin-boundary aliasing for lmdb/dict).
            for cycle in range(self._n_cycles):
                print(
                    f"[{backend}] cycle {cycle + 1}/{self._n_cycles} "
                    f"({len(goals)} goals)"
                )
                for i, (goal_type, goal_joints) in enumerate(goals):
                    self._do_round_trip(
                        cache=cache,
                        goal_type=goal_type,
                        goal_joints=goal_joints,
                        backend=backend,
                        phase="cycle",
                        cycle=cycle,
                        goal_idx=i,
                    )

    # ---------------------------------------------------------------
    # CSV output + entry point
    # ---------------------------------------------------------------

    def _write_csv(self) -> None:
        print(f"Writing {len(self._rows)} rows to {self._output_csv}")
        with open(self._output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)

    def run(self) -> None:
        goals = self._gen_goals()
        try:
            for backend in self._backends:
                try:
                    self._run_backend(backend, goals)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[{backend}] aborted with unexpected error: "
                        f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    )
        finally:
            self._write_csv()


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-contained trajectory-cache backend benchmark."
    )
    p.add_argument(
        "--backends",
        nargs="+",
        default=list(_VALID_BACKENDS),
        choices=_VALID_BACKENDS,
        help="Backends to benchmark (default: all four).",
    )
    p.add_argument("--n-joints", type=int, default=6)
    p.add_argument("--n-unique-goals", type=int, default=5000)
    p.add_argument("--n-cycles", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--joint-offset-range", type=float, default=0.3)
    p.add_argument("--robot-state-tolerance", type=float, default=0.05)
    p.add_argument("--position-tolerance", type=float, default=0.01)
    p.add_argument("--orientation-tolerance", type=float, default=0.05)
    p.add_argument("--max-trajectories", type=int, default=1)
    p.add_argument(
        "--output-csv", type=str, default="results/cache_benchmark_direct.csv"
    )
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    DirectCacheBenchmark(
        backends=args.backends,
        n_joints=args.n_joints,
        n_unique_goals=args.n_unique_goals,
        n_cycles=args.n_cycles,
        seed=args.seed,
        joint_offset_range=args.joint_offset_range,
        robot_state_tolerance=args.robot_state_tolerance,
        position_tolerance=args.position_tolerance,
        orientation_tolerance=args.orientation_tolerance,
        max_trajectories=args.max_trajectories,
        output_csv=args.output_csv,
        cache_dir=args.cache_dir,
    ).run()


if __name__ == "__main__":
    main()
