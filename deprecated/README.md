# Deprecated / reference-only

These files are kept for reference but are **not** part of the
self-contained library and will not run here — they depend on the
private robotics stack (ROS 2, MoveIt, `tabletop_rig`, `tabletop_tasks`)
that the rest of this repo was deliberately decoupled from.

| File | What it was | Why it's here |
| ---- | ----------- | ------------- |
| `requests.py` | Pydantic `PlanRequest` / `ConcatPlanRequest` models for the MoveIt planning interface. | Superseded by the library-agnostic `nns_trajectory_cache.types.TrajectoryCacheKey`. Tied to `geometry_msgs`, `moveit`, `moveit_msgs`, `pydantic`. |
| `cache_benchmark.py` | The *planning* benchmark: plans & executes `idle -> goal -> idle` round-trips through a live `Commander` / MoveIt and measures real plan + execute time and cache hit rate. | Inherently needs a running robot/planner, so it can't be made standalone. The self-contained sibling that measures pure cache query/update time is `src/nns_trajectory_cache/cache_benchmark_direct.py`. |
| `cache_benchmark.yaml`, `cache_benchmark_direct.yaml` | Task configs that launched the two benchmarks inside the `tabletop_tasks` runner. | The standalone benchmark is configured via CLI flags instead (see its module docstring). |

To map the old API onto the new one: a MoveIt `PlanRequest` becomes a
`TrajectoryCacheKey` (start joint positions + a `JointGoal` or
`CartesianGoal`), and a `RobotTrajectory` becomes whatever picklable
payload you wrap in a `TrajectoryCacheValue` along with its `path_cost`.
