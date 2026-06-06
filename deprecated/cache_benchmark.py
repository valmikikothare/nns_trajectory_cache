"""Trajectory cache backend benchmark task.

For each configured backend (`lmdb`, `dict`, `linear`, `kdtree`) this
task:

1. Swaps the manipulator's `_trajectory_cache` for a fresh instance of
   that backend (optionally wiping any on-disk file first so the
   backend starts cold).
2. Generates random joint-space goals around the `idle` state and
   plans+executes the round-trip `idle -> goal -> idle` for each.
   Goals where either leg fails to plan/execute are dropped from the
   "successful" list.
3. Once `n_unique_goals` successful round-trips have been collected,
   replays the same successful goal sequence for `n_cycles` more
   cycles. Subsequent cycles should hit the cache for every leg if
   the backend works correctly.

Per-leg, the task records: planning/cache-query time, execution time,
whether the result was retrieved from the cache (`cache_kwargs is None`
on return from `interface.plan(...)`), and success/error status.

All rows are appended to a CSV at `output_csv`.

This is a private-state benchmark: it directly reads
`commander._manipulators[robot_name]` and reassigns its
`_trajectory_cache`. Don't run it on the same path as the production
cache — pass a separate `cache_path_template` per backend.
"""

import asyncio
import csv
import os
import time
import traceback
from typing import Any, Optional

import numpy as np
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_model import (  # type: ignore[reportMissingModuleSource]
    RobotModel,
)
from moveit.core.robot_state import (  # type: ignore[reportMissingModuleSource]
    RobotState,
)
from tabletop_rig.exceptions import (
    ExecutionError,
    MoveitRecoverableError,
    PlanningError,
    TrajectoryError,
)
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

from tabletop_tasks.tasks.base import BaseTask

# Exception classes that count as "this goal failed; skip it".
# Anything else propagates and stops the benchmark.
_GOAL_FAILURE = (
    PlanningError,
    ExecutionError,
    MoveitRecoverableError,
    TrajectoryError,
    asyncio.TimeoutError,
)

_VALID_BACKENDS = ("lmdb", "dict", "linear", "kdtree")

_CSV_FIELDS = (
    "backend",
    "phase",
    "cycle",
    "goal_idx",
    "goal_type",
    "direction",
    "from_cache",
    "cache_size",
    "plan_time_s",
    "exec_time_s",
    "success",
    "error",
)

_MAX_GOAL_GEN_ATTEMPTS = 100


class CacheBenchmarkTask(BaseTask):
    """Benchmark four trajectory-cache backends head-to-head.

    Args:
        commander: The shared Commander.
        robot_name: Which arm to benchmark, e.g. "left_manipulator".
        backends: Subset of {"lmdb", "dict", "linear", "kdtree"} to
            run, in the given order.
        n_unique_goals: Number of distinct goals to successfully
            execute round-trip (`idle -> goal -> idle`) per backend
            before replaying.
        n_cycles: Number of additional replay cycles over the
            successful-goal sequence.
        max_failed_goals: Cap on random goals tried while building
            the successful-goal sequence. Prevents infinite loops if
            the joint range is too wide and most goals are
            unreachable.
        seed: PRNG seed for goal generation. Same seed gives the same
            goal sequence across backends so they're comparable.
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
        planning_pipeline / planning_time / max_attempts: Pass-through
            to `PlanRequest`. `None` keeps the interface's defaults.
        cache_kwargs_overrides: Per-test overrides applied on top of
            the manipulator's existing `trajectory_cache.kwargs`
            (tolerances, sort_by, max_trajectories). Useful for
            isolating the benchmark from production cache config.
    """

    def __init__(
        self,
        commander: Commander,
        *,
        robot_name: str = "left_manipulator",
        backends: Optional[list[str]] = None,
        n_unique_goals: int = 20,
        n_cycles: int = 3,
        max_goal_gen_attempts: int = 200,
        max_failed_goals: int = 200,
        seed: int = 42,
        joint_offset_range_radians: float = 1.0,
        output_csv: str = "$TABLETOP_CACHE_DIR/cache_benchmark.csv",
        cache_path_template: str = (
            "$TABLETOP_CACHE_DIR/trajectory_cache/benchmark_{robot}_{backend}"
        ),
        wipe_cache_before_run: bool = True,
        planning_pipeline: Optional[str] = None,
        planning_time: Optional[float] = None,
        max_attempts: Optional[int] = None,
        cache_kwargs_overrides: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__("cache_benchmark", commander)

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

        if max_goal_gen_attempts < 1:
            raise ValueError("'max_goal_gen_attempts' must be at least 0")
        self._max_goal_gen_attempts = max_goal_gen_attempts

        if max_failed_goals < 0:
            raise ValueError("'max_failed_goals' must be at least 0")
        self._max_failed_goals = max_failed_goals

        self._seed = seed
        self._joint_offset_range = float(joint_offset_range_radians)
        if self._joint_offset_range <= 0:
            raise ValueError("'joint_offset_range_radians' must be positive")

        self._output_csv = os.path.expandvars(os.path.expanduser(output_csv))
        os.makedirs(os.path.dirname(self._output_csv), exist_ok=True)

        self._cache_path_template = cache_path_template
        self._wipe_cache_before_run = wipe_cache_before_run

        self._planning_pipeline = planning_pipeline
        self._planning_time = planning_time
        self._max_attempts = max_attempts

        self._cache_kwargs_overrides = dict(cache_kwargs_overrides or {})

        # All rows accumulated across backends, flushed at the end.
        self._rows: list[dict[str, Any]] = []

    @property
    def _interface(self):
        """The underlying ObjectManipulationInterface (private access)."""
        return self.commander._manipulators[self._robot_name]

    # ---------------------------------------------------------------
    # Cache construction
    # ---------------------------------------------------------------

    def _resolve_cache_path(self, backend: str) -> str:
        path = self._cache_path_template.format(
            robot=self._robot_name, backend=backend
        )
        path = os.path.expandvars(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _wipe_path(self, path: str) -> None:
        # LMDB also leaves a `.lock` sibling that we have to remove.
        for filepath in (path, path + "-lock"):
            if os.path.exists(filepath):
                os.remove(filepath)
                self.log(f"Removed stale cache file: {filepath}")

    def _build_cache(self, backend: str) -> TrajectoryCache:
        interface = self._interface
        moveit = self.commander._moveit

        # Start from the manipulator's configured cache kwargs (per-arm
        # path override + common tolerances), then layer the
        # benchmark-specific overrides on top. Replace `path` with our
        # per-backend file.
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
    ) -> RobotState | PoseStamped:
        """Build a RobotState by perturbing every joint of idle uniformly."""
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
                return pose_stamped_msg(pose=pose, frame_id=frame_id)
            else:
                return state

        raise RuntimeError(
            f"Could not generate valid goal state in {_MAX_GOAL_GEN_ATTEMPTS} attempts"
        )

    def _gen_goals(self) -> list[RobotState | PoseStamped]:
        rng = np.random.default_rng(self._seed)

        goals: list[RobotState | PoseStamped] = []
        for _ in range(self._n_unique_goals):
            goals.append(self._random_goal(rng))

        return goals

    # ---------------------------------------------------------------
    # Plan + execute a single leg
    # ---------------------------------------------------------------

    def _make_request(self, goal: RobotState | PoseStamped) -> PlanRequest:
        kwargs: dict[str, Any] = {"goal": goal}
        if self._planning_pipeline is not None:
            kwargs["planning_pipeline"] = self._planning_pipeline
        if self._planning_time is not None:
            kwargs["planning_time"] = self._planning_time
        if self._max_attempts is not None:
            kwargs["max_attempts"] = self._max_attempts
        return PlanRequest(**kwargs)

    async def _plan_and_execute_leg(
        self,
        goal: RobotState | PoseStamped,
        backend: str,
        phase: str,
        cycle: int,
        goal_idx: int,
        direction: str,
    ) -> bool:
        """Plan and execute one leg; record a row; return True on success."""
        interface = self._interface
        request = self._make_request(goal)

        goal_type = "robot_state" if isinstance(goal, RobotState) else "pose"
        plan_time = float("nan")
        exec_time = float("nan")
        from_cache: Optional[bool] = None
        cache_size: int = -1
        success = False
        error = ""

        try:
            cache_size = len(interface._trajectory_cache)
            t0 = time.perf_counter()
            trajectory, cache_kwargs = await interface.plan(request=request)
            plan_time = time.perf_counter() - t0
            from_cache = cache_kwargs is None

            t0 = time.perf_counter()
            await interface.execute(trajectory)
            exec_time = time.perf_counter() - t0

            # Save the freshly-planned trajectory under the same path
            # the production code uses (`cache_trajectories` no-ops
            # when the trajectory came from cache).
            if cache_kwargs is not None:
                interface.cache_trajectories(cache_kwargs)

            success = True
        except _GOAL_FAILURE as e:
            error = f"{type(e).__name__}: {e}"
            self.log(
                f"[{backend}] {phase} cycle={cycle} goal={goal_idx} "
                f"{direction} failed: {error}",
                severity="WARN",
            )
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

        self._rows.append(
            {
                "backend": backend,
                "phase": phase,
                "cycle": cycle,
                "goal_idx": goal_idx,
                "goal_type": goal_type,
                "direction": direction,
                "from_cache": from_cache,
                "cache_size": cache_size,
                "plan_time_s": (
                    "" if np.isnan(plan_time) else f"{plan_time:.6f}"
                ),
                "exec_time_s": (
                    "" if np.isnan(exec_time) else f"{exec_time:.6f}"
                ),
                "success": success,
                "error": error,
            }
        )
        return success

    async def _do_round_trip(
        self,
        goal: RobotState | PoseStamped,
        backend: str,
        phase: str,
        cycle: int,
        goal_idx: int,
    ) -> bool:
        """idle -> goal -> idle. Returns True only if both legs succeeded."""
        idle = self._idle_state()
        ok_out = await self._plan_and_execute_leg(
            goal=goal,
            backend=backend,
            phase=phase,
            cycle=cycle,
            goal_idx=goal_idx,
            direction="to_goal",
        )
        if not ok_out:
            # Try to get back to a known state for the next attempt.
            await self._best_effort_return_to_idle(
                idle, backend, phase, cycle, goal_idx
            )
            return False
        ok_back = await self._plan_and_execute_leg(
            goal=idle,
            backend=backend,
            phase=phase,
            cycle=cycle,
            goal_idx=goal_idx,
            direction="to_idle",
        )
        return ok_back

    async def _best_effort_return_to_idle(
        self,
        idle: RobotState,
        backend: str,
        phase: str,
        cycle: int,
        goal_idx: int,
    ) -> None:
        """After a failed leg, try to plan back to idle so we don't leave
        the robot stranded mid-trajectory. Failures here are logged but
        not propagated."""
        try:
            await self._plan_and_execute_leg(
                goal=idle,
                backend=backend,
                phase=phase,
                cycle=cycle,
                goal_idx=goal_idx,
                direction="recover_to_idle",
            )
        except Exception as e:
            self.log(
                f"[{backend}] best-effort return-to-idle failed: "
                f"{type(e).__name__}: {e}",
                severity="WARN",
            )

    # ---------------------------------------------------------------
    # Per-backend runner
    # ---------------------------------------------------------------

    async def _run_backend(
        self, backend: str, goals: list[RobotState | PoseStamped]
    ) -> None:
        self.log(f"=== Benchmark backend: {backend} ===")
        self._swap_cache(backend)

        # `manipulation_context` locks arms, occludes smartglass, and
        # resets the manipulator's state machine to IDLE. We then
        # bypass the state machine by calling `interface.plan` and
        # `interface.execute` directly — safe because they still
        # check `safe_to_execute` and trajectory start tolerance.
        async with self.commander.manipulation_context(self._robot_name):
            successful_goals: list[tuple[int, RobotState | PoseStamped]] = []
            for i, goal in enumerate(goals):
                ok = await self._do_round_trip(
                    goal=goal,
                    backend=backend,
                    phase="collect",
                    cycle=0,
                    goal_idx=i,
                )
                if ok:
                    successful_goals.append((i, goal))
                if i % 100 == 0:
                    self.log(f"[{backend}] collect progress: {i}/{len(goals)}")

            self.log(
                f"[{backend}] collected {len(successful_goals)} "
                f"successful goals out of {len(goals)} attempts"
            )

            if len(successful_goals) == 0:
                self.log(
                    f"[{backend}] no successful goals — skipping cycle phase",
                    severity="WARN",
                )
                return

            for cycle in range(self._n_cycles):
                self.log(
                    f"[{backend}] cycle {cycle + 1}/{self._n_cycles} "
                    f"({len(successful_goals)} goals)"
                )
                for i, goal in successful_goals:
                    await self._do_round_trip(
                        goal=goal,
                        backend=backend,
                        phase="cycle",
                        cycle=cycle,
                        goal_idx=i,
                    )
                    if i % 100 == 0:
                        self.log(
                            f"[{backend}] cycle progress: {i}/{len(successful_goals)}"
                        )

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
            # Always flush whatever rows we collected, even on abort.
            self._write_csv()
