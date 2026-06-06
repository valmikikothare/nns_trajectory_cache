"""Library-agnostic data model for the trajectory cache.

The cache is keyed on a :class:`TrajectoryCacheKey` and valued on a
:class:`TrajectoryCacheValue`. Neither type knows anything about ROS,
MoveIt, or any particular trajectory representation — the user converts
whatever planning request and trajectory they have into these plain
dataclasses before handing them to the cache, and converts back on the
way out.

A key is a *start configuration* plus a *goal*, where the goal is
either:

- :class:`JointGoal` — a target joint configuration (joint-space goal),
  or
- :class:`CartesianGoal` — a target end-effector pose (Cartesian goal).

A value wraps an opaque, picklable ``trajectory`` payload together with
a scalar ``path_cost`` used to rank competing trajectories (lower is
better). The payload is whatever the caller wants to store — a
serialized message, a numpy array, a list of waypoints — the cache
never inspects it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class JointGoal:
    """A joint-space goal: a target configuration keyed by joint name.

    Attributes:
        joint_positions: Target position (radians/metres) per joint
            name. Must cover the same joints used everywhere else in
            the cache for a given group.
    """

    joint_positions: Mapping[str, float]


@dataclass(frozen=True)
class CartesianGoal:
    """A Cartesian goal: a target end-effector pose.

    Attributes:
        position: ``(x, y, z)`` translation in metres.
        orientation: ``(x, y, z, w)`` unit quaternion.
        frame_id: The reference frame the pose is expressed in. Must
            match the cache's configured ``planning_frame``.
    """

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    frame_id: str


Goal = JointGoal | CartesianGoal
"""A planning goal: either a joint-space or a Cartesian target."""


@dataclass
class TrajectoryCacheKey:
    """A library-agnostic motion-planning request fingerprint.

    This is everything the cache needs to index a request: the start
    configuration, the goal (joint-space or Cartesian), and the group /
    pose-link metadata that namespaces it. It deliberately omits planner
    settings, scene data, constraints, etc. — those don't affect which
    cached trajectory satisfies the request.

    Attributes:
        start_joint_positions: Start configuration, position per joint
            name.
        goal: The target, as a :class:`JointGoal` or
            :class:`CartesianGoal`.
        group_name: The joint group this request addresses. Must equal
            the cache's configured ``group_name``.
        pose_link: For Cartesian goals, the end-effector link the pose
            is expressed against; must equal the cache's configured
            ``pose_link``. Must be ``None`` for joint-space goals.
    """

    start_joint_positions: Mapping[str, float]
    goal: Goal
    group_name: str
    pose_link: Optional[str] = None


@dataclass(frozen=True, eq=False)
class TrajectoryCacheValue:
    """A cached trajectory plus its sortable path cost.

    The cache keeps competing values for a match ranked by ``path_cost``
    (lower is better) and returns them cheapest-first. ``trajectory`` is
    an opaque payload supplied by the caller; it must be picklable for
    the persistent backends (LMDB, and the on-disk dict / linear /
    kdtree stores) but is otherwise never inspected by the cache.

    Attributes:
        trajectory: Opaque, picklable trajectory payload.
        path_cost: Scalar cost used to rank trajectories (e.g. path
            length or duration). Lower is better. Must be finite and
            non-negative.
        metadata: Optional free-form, picklable metadata stored
            alongside the trajectory (ignored by the cache).
    """

    trajectory: Any
    path_cost: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cost = float(self.path_cost)
        if cost != cost or cost in (float("inf"), float("-inf")):
            raise ValueError(f"path_cost must be finite: {self.path_cost!r}")
        if cost < 0:
            raise ValueError(f"path_cost must be non-negative: {cost}")
        # Normalize to a float without tripping the frozen guard.
        object.__setattr__(self, "path_cost", cost)

    def __lt__(self, other: "TrajectoryCacheValue") -> bool:
        return self.path_cost < other.path_cost
