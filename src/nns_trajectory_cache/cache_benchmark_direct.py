"""Direct-access trajectory-cache benchmark task.

A stripped-down sibling of `CacheBenchmarkTask` that exercises the
cache backends *without* invoking MoveIt's planning or trajectory
execution at all.

For each configured backend (`lmdb`, `dict`, `linear`, `kdtree`) this
task:

1. Swaps the manipulator's `_trajectory_cache` for a fresh instance of
   that backend (optionally wiping any on-disk file first so the
   backend starts cold).
2. Generates random joint-space goals around the `idle` state.
3. For every leg (`idle -> goal` and `goal -> idle`):
   - Builds the matching `PlanRequest`.
   - Times a cache query — `cache.get_trajectories(request)`.
   - On miss, builds a two-waypoint *dummy* trajectory whose start /
     end states exactly match `request.start_state` and `request.goal`
     (so it passes `_validate_trajectory_quality`), then times
     `cache.cache_trajectory(...)`.

Because no plan/execute round-trip is involved, the cache saturates
orders of magnitude faster than `CacheBenchmarkTask`. This lets us
push `n_unique_goals` into the tens of thousands and measure how
*query time* and *update time* scale with cache size for each backend.

Per leg we record query/update times, hit-vs-miss, and the cache size
both before and after any insert. Rows are appended to a CSV at
`output_csv` distinct from the planning benchmark's output.

This task does not lock arms or enter `manipulation_context` because
nothing is executed; it only mutates `interface._trajectory_cache` in
place. Don't run it on the same path as the production cache — pass a
separate `cache_path_template` per backend.
"""

import asyncio
import csv
import os
import time
import traceback
from typing import Any, Optional

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_model import (  # type: ignore[reportMissingModuleSource]
    RobotModel,
)
from moveit.core.robot_state import (  # type: ignore[reportMissingModuleSource]
    RobotState,
)
from moveit.core.robot_trajectory import (  # type: ignore[reportMissingModuleSource]
    RobotTrajectory,
)
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from tabletop_rig.interfaces.moveit.requests import PlanRequest
from tabletop_rig.interfaces.moveit.trajectory_cache import TrajectoryCache
from tabletop_rig.interfaces.moveit.trajectory_cache_dict import (
    DictFuzzyTrajectoryCache,
)
from tabletop_rig.interfaces.moveit.trajectory_cache_kdtree import (
    KDTreeTrajectoryCache,
)
from tabletop_rig.interfaces.moveit.trajectory_cache_linear import (
    LinearTrajectoryCache,
)
from tabletop_rig.interfaces.moveit.trajectory_cache_lmdb import (
    LMDBFuzzyTrajectoryCache,
)
from tabletop_rig.nodes import Commander
from tabletop_rig.utils.ros import pose_stamped_msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tabletop_tasks.tasks.base import BaseTask

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

_MAX_GOAL_GEN_ATTEMPTS = 100

# Dummy trajectory duration. Must be > 0 so `sort_by="path_duration"`
# can accept it (see `TrajectoryCacheValue.__init__`).
_DUMMY_TRAJ_DURATION_S = 1.0


class CacheBenchmarkDirectTask(BaseTask):
    """Benchmark trajectory-cache backends via direct query/update.

    Bypasses MoveIt planning and trajectory execution entirely. Every
    miss is satisfied with a synthetic two-waypoint trajectory built
    to exactly match the request's start/goal endpoints, so the cache
    receives well-formed entries without paying for a planner call.

    Args:
        commander: The shared Commander.
        robot_name: Which arm to benchmark, e.g. "left_manipulator".
        backends: Subset of {"lmdb", "dict", "linear", "kdtree"} to
            run, in the given order.
        n_unique_goals: Number of distinct random goals to generate
            per backend. Each goal contributes one round-trip
            (`idle -> goal -> idle`).
        n_cycles: Number of additional replay cycles over the goal
            sequence (parity with `CacheBenchmarkTask`; with the
            cache pre-populated these become pure-hit measurements).
        seed: PRNG seed for goal generation. Same seed produces the
            same goal sequence across backends so timings are
            directly comparable.
        joint_offset_range_radians: Each joint is sampled uniformly
            from `idle ± this`.
        output_csv: Where to write the per-leg results. `~` and
            `$VARS` are expanded.
        cache_path_template: Format string for the cache persistence
            file. `{robot}` and `{backend}` placeholders are
            substituted. Each backend writes to its own file.
        wipe_cache_before_run: If True (default), delete any existing
            cache file for the backend before constructing the cache
            so every backend benchmark starts cold.
        cache_kwargs_overrides: Per-test overrides applied on top of
            the manipulator's existing `trajectory_cache.kwargs`
            (tolerances, sort_by, max_trajectories).
    """

    def __init__(
        self,
        commander: Commander,
        *,
        robot_name: str = "left_manipulator",
        backends: Optional[list[str]] = None,
        n_unique_goals: int = 10000,
        n_cycles: int = 1,
        seed: int = 42,
        joint_offset_range_radians: float = 1.0,
        output_csv: str = "$TABLETOP_CACHE_DIR/cache_benchmark_direct.csv",
        cache_path_template: str = (
            "$TABLETOP_CACHE_DIR/trajectory_cache/"
            "benchmark_direct_{robot}_{backend}"
        ),
        wipe_cache_before_run: bool = True,
        cache_kwargs_overrides: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__("cache_benchmark_direct", commander)

        if robot_name not in commander._manipulators:
            raise ValueError(
                f"Unknown robot_name {robot_name!r}; available: "
                f"{list(commander._manipulators.keys())}"
            )
        self._robot_name = robot_name

        backends = list(backends) if backends else list(_VALID_BACKENDS)
        unknown = set(backends) - set(_VALID_BACKENDS)
        if unknown:
            raise ValueError(
                f"Unknown cache backends: {sorted(unknown)}. "
                f"Expected one of: {_VALID_BACKENDS}"
            )
        self._backends = backends

        if n_unique_goals < 1:
            raise ValueError("'n_unique_goals' must be at least 1")
        self._n_unique_goals = n_unique_goals

        if n_cycles < 0:
            raise ValueError("'n_cycles' must be non-negative")
        self._n_cycles = n_cycles

        self._seed = seed
        self._joint_offset_range = float(joint_offset_range_radians)
        if self._joint_offset_range <= 0:
            raise ValueError("'joint_offset_range_radians' must be positive")

        self._output_csv = os.path.expandvars(os.path.expanduser(output_csv))
        os.makedirs(os.path.dirname(self._output_csv), exist_ok=True)

        self._cache_path_template = cache_path_template
        self._wipe_cache_before_run = wipe_cache_before_run

        self._cache_kwargs_overrides = dict(cache_kwargs_overrides or {})

        self._rows: list[dict[str, Any]] = []

    @property
    def _interface(self):
        """The underlying ObjectManipulationInterface (private access)."""
        return self.commander._manipulators[self._robot_name]

    # ---------------------------------------------------------------
    # Cache construction (mirrors CacheBenchmarkTask)
    # ---------------------------------------------------------------

    def _resolve_cache_path(self, backend: str) -> str:
        path = self._cache_path_template.format(
            robot=self._robot_name, backend=backend
        )
        path = os.path.expandvars(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _wipe_path(self, path: str) -> None:
        for filepath in (path, path + "-lock"):
            if os.path.exists(filepath):
                os.remove(filepath)
                self.log(f"Removed stale cache file: {filepath}")

    def _build_cache(self, backend: str) -> TrajectoryCache:
        interface = self._interface
        moveit = self.commander._moveit

        cache_kwargs: dict[str, Any] = dict(
            interface.param("trajectory_cache.kwargs")
        )
        cache_kwargs.update(self._cache_kwargs_overrides)

        cache_kwargs["path"] = self._resolve_cache_path(backend)
        if self._wipe_cache_before_run:
            self._wipe_path(cache_kwargs["path"])

        common_kwargs: dict[str, Any] = dict(
            scene_hash=moveit.scene_hash(include_robot=True),
            planning_frame=moveit.planning_frame,
            group_name=interface.group_name,
            pose_link=interface.default_pose_link,
            parent_logger=interface.get_logger(),
            **cache_kwargs,
        )

        match backend:
            case "lmdb":
                return LMDBFuzzyTrajectoryCache(**common_kwargs)
            case "dict":
                return DictFuzzyTrajectoryCache(**common_kwargs)
            case "linear":
                return LinearTrajectoryCache(**common_kwargs)
            case "kdtree":
                return KDTreeTrajectoryCache(
                    sample_state=moveit.get_current_state(),
                    **common_kwargs,
                )
            case _:
                raise AssertionError(f"unreachable backend: {backend!r}")

    def _swap_cache(self, backend: str) -> TrajectoryCache:
        """Close the current cache and install a fresh one for `backend`."""
        interface = self._interface
        old_cache = interface._trajectory_cache
        try:
            old_cache.close()
        except Exception as e:
            self.log(
                f"Error closing previous cache ({type(old_cache).__name__}): "
                f"{e}",
                severity="WARN",
            )

        new_cache = self._build_cache(backend)
        new_cache.open()
        interface._trajectory_cache = new_cache
        self.log(
            f"Installed {type(new_cache).__name__} cache at "
            f"{getattr(new_cache, 'path', None)}"
        )
        return new_cache

    # ---------------------------------------------------------------
    # Goal generation
    # ---------------------------------------------------------------

    def _idle_state(self) -> RobotState:
        return self.commander._moveit.get_target_state(
            "idle", self._interface.group_name
        )

    def _random_goal(
        self, rng: np.random.Generator
    ) -> tuple[RobotState | PoseStamped, RobotState]:
        """Sample one valid goal.

        Returns the goal (either a `RobotState` or a `PoseStamped`)
        and the underlying `RobotState` that produced it. For pose
        goals we keep this state around because the dummy trajectory's
        end-waypoint joints must round-trip to the same FK pose; only
        the state that *generated* the pose is guaranteed to.
        """
        for _ in range(_MAX_GOAL_GEN_ATTEMPTS):
            state = self._idle_state()
            positions = dict(state.joint_positions)
            joint_names = self.commander._moveit.get_joint_names(
                self._interface.group_name
            )
            offsets = rng.uniform(
                -self._joint_offset_range,
                self._joint_offset_range,
                size=len(joint_names),
            )
            for joint, offset in zip(joint_names, offsets):
                positions[joint] = positions[joint] + float(offset)

            state.joint_positions = positions
            state.update()
            if not self.commander._moveit.is_state_valid(
                state, group_name=self._robot_name, verbose=False
            ):
                continue

            return_pose: bool = rng.choice([True, False])
            if return_pose:
                link = self._interface.default_pose_link
                pose = state.get_pose(link)
                model: RobotModel = state.robot_model
                frame_id = model.model_frame
                assert frame_id == self.commander._moveit.planning_frame
                return pose_stamped_msg(pose=pose, frame_id=frame_id), state
            else:
                return state, state

        raise RuntimeError(
            f"Could not generate valid goal state in "
            f"{_MAX_GOAL_GEN_ATTEMPTS} attempts"
        )

    def _gen_goals(
        self,
    ) -> list[tuple[RobotState | PoseStamped, RobotState]]:
        rng = np.random.default_rng(self._seed)
        return [self._random_goal(rng) for _ in range(self._n_unique_goals)]

    # ---------------------------------------------------------------
    # Dummy trajectory construction
    # ---------------------------------------------------------------

    def _make_dummy_trajectory(
        self, start_state: RobotState, end_state: RobotState
    ) -> RobotTrajectory:
        """Build a 2-waypoint trajectory that exactly bridges start/end.

        The trajectory's joint_names are taken from the configured
        group's `active_joint_model_names` and positions are reordered
        to match — `_validate_trajectory_quality` resolves endpoint
        joints by name via `get_joint_group_positions`, so any name/
        position ordering mismatch silently corrupts the endpoints.

        Duration of 1s ensures `sort_by="path_duration"` accepts it.
        """
        group_name = self._interface.group_name
        joint_names: list[str] = self.commander._moveit.get_joint_names(
            group_name
        )

        msg = RobotTrajectoryMsg()
        jt = JointTrajectory()
        jt.joint_names = list(joint_names)

        start_positions = start_state.joint_positions
        end_positions = end_state.joint_positions

        p0 = JointTrajectoryPoint()
        p0.positions = [float(start_positions[j]) for j in joint_names]
        p0.time_from_start = Duration(sec=0, nanosec=0)

        p1 = JointTrajectoryPoint()
        p1.positions = [float(end_positions[j]) for j in joint_names]
        dur_ns = int(_DUMMY_TRAJ_DURATION_S * 1e9)
        p1.time_from_start = Duration(
            sec=int(_DUMMY_TRAJ_DURATION_S),
            nanosec=dur_ns - int(_DUMMY_TRAJ_DURATION_S) * int(1e9),
        )

        jt.points = [p0, p1]
        msg.joint_trajectory = jt

        # `set_robot_trajectory_msg` needs a clean reference state to
        # populate non-group joints; it also drops `joint_model_group_name`,
        # so re-assign after.
        reference_state = self._idle_state()
        if reference_state.dirty:
            reference_state.update()
        trajectory = RobotTrajectory(reference_state.robot_model)
        trajectory.set_robot_trajectory_msg(reference_state, msg)
        trajectory.joint_model_group_name = group_name
        return trajectory

    # ---------------------------------------------------------------
    # Single leg: query, then update on miss
    # ---------------------------------------------------------------

    def _process_leg(
        self,
        *,
        start_state: RobotState,
        goal: RobotState | PoseStamped,
        end_state: RobotState,
        backend: str,
        phase: str,
        cycle: int,
        goal_idx: int,
        direction: str,
    ) -> bool:
        """Query the cache; on miss insert a dummy trajectory. Record a row."""
        interface = self._interface
        cache = interface._trajectory_cache

        request_kwargs: dict[str, Any] = {
            "start_state": start_state,
            "goal": goal,
            "group_name": interface.group_name,
        }
        if isinstance(goal, PoseStamped):
            request_kwargs["pose_link"] = interface.default_pose_link
        request = PlanRequest(**request_kwargs)

        goal_type = "robot_state" if isinstance(goal, RobotState) else "pose"
        hit: Optional[bool] = None
        query_time = float("nan")
        update_time = float("nan")
        cache_size_after: Optional[int] = None
        success = False
        error = ""

        try:
            cache_size_before = len(cache)

            t0 = time.perf_counter()
            try:
                cache.get_trajectories(request)
                hit = True
            except KeyError:
                hit = False
            query_time = time.perf_counter() - t0

            if not hit:
                trajectory = self._make_dummy_trajectory(
                    start_state, end_state
                )
                t0 = time.perf_counter()
                cache.cache_trajectory(trajectory, request=request)
                update_time = time.perf_counter() - t0

            cache_size_after = len(cache)
            success = True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            self.log(
                f"[{backend}] {phase} cycle={cycle} goal={goal_idx} "
                f"{direction} unexpected: {error}\n"
                f"{''.join(traceback.format_exc())}",
                severity="ERROR",
            )
            cache_size_before = -1
            cache_size_after = -1

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
                "query_time_s": (
                    "" if np.isnan(query_time) else f"{query_time:.9f}"
                ),
                "update_time_s": (
                    "" if np.isnan(update_time) else f"{update_time:.9f}"
                ),
                "success": success,
                "error": error,
            }
        )
        return success

    def _do_round_trip(
        self,
        *,
        goal: RobotState | PoseStamped,
        end_state: RobotState,
        backend: str,
        phase: str,
        cycle: int,
        goal_idx: int,
    ) -> None:
        """idle -> goal -> idle (cache-only; no execution)."""
        idle = self._idle_state()
        # Forward leg: start=idle, goal=goal (RobotState or pose).
        # End-waypoint joints come from `end_state` so a pose goal's
        # FK round-trips.
        self._process_leg(
            start_state=idle,
            goal=goal,
            end_state=end_state,
            backend=backend,
            phase=phase,
            cycle=cycle,
            goal_idx=goal_idx,
            direction="to_goal",
        )
        # Return leg: start=end_state, goal=idle (always RobotState).
        self._process_leg(
            start_state=end_state,
            goal=idle,
            end_state=idle,
            backend=backend,
            phase=phase,
            cycle=cycle,
            goal_idx=goal_idx,
            direction="to_idle",
        )

    # ---------------------------------------------------------------
    # Per-backend runner
    # ---------------------------------------------------------------

    async def _run_backend(
        self,
        backend: str,
        goals: list[tuple[RobotState | PoseStamped, RobotState]],
    ) -> None:
        self.log(f"=== Direct cache benchmark backend: {backend} ===")
        self._swap_cache(backend)

        # No `manipulation_context`: nothing executes, so no need to
        # lock arms / occlude smartglass / move the state machine.

        # Collect phase — first-pass through every goal. Cache grows
        # roughly monotonically here so this gives clean
        # time-vs-cache-size data.
        for i, (goal, end_state) in enumerate(goals):
            self._do_round_trip(
                goal=goal,
                end_state=end_state,
                backend=backend,
                phase="collect",
                cycle=0,
                goal_idx=i,
            )
            # Yield to the event loop occasionally so the node stays
            # responsive on very long runs.
            if i % 100 == 0:
                self.log(f"[{backend}] collect progress: {i}/{len(goals)}")
                await asyncio.sleep(0)

        self.log(
            f"[{backend}] collect done, cache_size="
            f"{len(self._interface._trajectory_cache)}"
        )

        # Cycle phase — replays. With the dummy-trajectory inserts in
        # the collect phase, every leg here should be a hit (modulo
        # fuzzy-bin aliasing for the lmdb/dict backends).
        for cycle in range(self._n_cycles):
            self.log(
                f"[{backend}] cycle {cycle + 1}/{self._n_cycles} "
                f"({len(goals)} goals)"
            )
            for i, (goal, end_state) in enumerate(goals):
                self._do_round_trip(
                    goal=goal,
                    end_state=end_state,
                    backend=backend,
                    phase="cycle",
                    cycle=cycle,
                    goal_idx=i,
                )
                if i % 100 == 0:
                    self.log(f"[{backend}] cycle progress: {i}/{len(goals)}")
                    await asyncio.sleep(0)

    # ---------------------------------------------------------------
    # CSV output
    # ---------------------------------------------------------------

    def _write_csv(self) -> None:
        self.log(f"Writing {len(self._rows)} rows to {self._output_csv}")
        with open(self._output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)  # pyright: ignore[reportArgumentType]

    # ---------------------------------------------------------------
    # Entry point
    # ---------------------------------------------------------------

    async def run(self) -> None:
        goals = self._gen_goals()
        try:
            for backend in self._backends:
                try:
                    await self._run_backend(backend, goals)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.log(
                        f"[{backend}] aborted with unexpected error: "
                        f"{type(e).__name__}: {e}\n"
                        f"{''.join(traceback.format_exc())}",
                        severity="ERROR",
                    )
        finally:
            self._write_csv()
