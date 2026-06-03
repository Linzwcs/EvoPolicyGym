"""Family-level task descriptions for bulk Gymnasium registrations."""

from __future__ import annotations

from dataclasses import dataclass

from .spec import TaskDoc


def task_doc(env_id: str, family: str) -> TaskDoc:
    """Return a useful fallback task document for one discovered env id."""

    template = _TEMPLATES.get(family, _GENERIC)
    return TaskDoc(
        title=env_id,
        objective=template.objective.format(env_id=env_id),
        observation=template.observation,
        action=template.action,
        reward=template.reward,
        notes=template.notes,
    )


@dataclass(frozen=True, slots=True)
class _Template:
    objective: str
    observation: str
    action: str
    reward: str
    notes: tuple[str, ...] = ()


_GENERIC = _Template(
    objective="Maximize total Gymnasium reward in `{env_id}`.",
    observation="Use `obs_space` for the exact observation structure. Values may be vectors, images, dictionaries, or nested Gymnasium spaces.",
    action="Return one JSON-safe action matching `action_space`. The adapter repairs invalid numeric actions where practical and records diagnostics in feedback.",
    reward="Episode return is the sum of Gymnasium rewards. Environment-specific score meaning follows the upstream task.",
    notes=("This is a dynamic bulk registration; prefer curated aliases when they exist for benchmark-critical tasks.",),
)

_TEMPLATES: dict[str, _Template] = {
    "Official Gymnasium: Classic Control": _Template(
        objective="Solve the classic-control task `{env_id}` by stabilizing or moving the low-dimensional physical system toward its goal.",
        observation="Usually a compact numeric vector describing positions, velocities, angles, or angular velocities. Use `obs_space` for exact fields and bounds.",
        action="Usually a discrete action or a small continuous control vector. Return values matching `action_space`.",
        reward="Rewards measure task progress, survival, or negative control cost depending on the upstream environment.",
    ),
    "Official Gymnasium: Toy Text": _Template(
        objective="Solve the symbolic decision process `{env_id}` using the visible discrete state and train feedback.",
        observation="Usually a discrete integer or tuple-like symbolic state. Decode it according to the upstream task when useful.",
        action="Usually a small Discrete action set for movement, card play, pickup/dropoff, or other symbolic operations.",
        reward="Rewards are often sparse or terminal, with penalties for illegal or inefficient moves.",
    ),
    "Official Gymnasium: Box2D": _Template(
        objective="Control the Box2D task `{env_id}` and maximize episode return under the declared step limit.",
        observation="May be a vector state or an RGB image. Image observations are stored externally in `observations.npy` when `obs_storage` is external.",
        action="May be Discrete or continuous Box control. Return JSON-safe values matching `action_space` bounds.",
        reward="Rewards combine progress, stability, task completion, and control penalties according to the upstream environment.",
        notes=("Box2D environments may be heavier than classic-control tasks and may require native optional dependencies.",),
    ),
    "Official Gymnasium: MuJoCo": _Template(
        objective="Control the MuJoCo body in `{env_id}` to maximize locomotion or manipulation return.",
        observation="Usually a high-dimensional numeric vector of joint positions, velocities, body states, or task-specific measurements.",
        action="Usually a continuous Box vector of actuator controls. Keep actions within the declared bounds.",
        reward="Rewards typically combine forward progress or goal achievement with health, stability, and control-cost terms.",
        notes=("MuJoCo environments are native-dependency tasks; smoke cost and compatibility vary by version.",),
    ),
    "Official Gymnasium: Atari / ALE": _Template(
        objective="Maximize game score in the Atari/ALE environment `{env_id}`.",
        observation="Usually image frames. Large frames are stored externally as episode observation artifacts when `obs_storage` is external.",
        action="Return a Discrete emulator action id from `action_space`. Action meanings are game-specific.",
        reward="Rewards are game score deltas or clipped/scaled variants depending on the upstream wrapper configuration.",
        notes=("Use feedback trajectories and image artifacts to infer game state, object positions, and failure modes.",),
    ),
    "MiniGrid": _Template(
        objective="Solve the MiniGrid mission in `{env_id}` by navigating, picking up objects, opening doors, or reaching goals as required.",
        observation="Usually a partial symbolic grid, optional RGB image, and/or mission text. Use the Dict schema keys to identify each component.",
        action="Usually a small Discrete action set for turning, moving, pickup/drop, toggle, and done.",
        reward="Rewards are usually sparse success rewards, often shaped by shorter episode length.",
        notes=("Mission text is part of the task state when exposed by the observation schema.",),
    ),
    "MiniWorld": _Template(
        objective="Navigate or interact in the MiniWorld 3D task `{env_id}`.",
        observation="Usually RGB first-person images and sometimes compact state fields. Image observations may be externalized in feedback artifacts.",
        action="Usually discrete or continuous movement and camera controls. Match `action_space` exactly.",
        reward="Rewards measure reaching goals, collecting objects, or completing navigation tasks.",
    ),
    "HighwayEnv": _Template(
        objective="Drive safely and effectively in the traffic scenario `{env_id}`.",
        observation="May be kinematic vehicle features, occupancy grids, or image-like scenario encodings depending on upstream configuration.",
        action="May be discrete high-level driving commands or continuous steering/acceleration controls.",
        reward="Rewards balance progress, speed, lane choice, comfort, collision avoidance, and task-specific safety terms.",
    ),
    "Gymnasium-Robotics": _Template(
        objective="Solve the robotics or goal-conditioned task `{env_id}`.",
        observation="Often a Dict with `observation`, `achieved_goal`, and `desired_goal` fields. Use each schema field explicitly.",
        action="Usually a continuous Box vector for robot controls or target deltas.",
        reward="Rewards usually measure distance to a goal or sparse task success; goal fields are central to policy decisions.",
        notes=("Do not assume hidden validation goals are visible beyond the observation fields supplied at runtime.",),
    ),
    "MO-Gymnasium": _Template(
        objective="Optimize the multi-objective task `{env_id}` under EvoPolicyGym's scalar episode-return reporting.",
        observation="Use `obs_space` for the exact state representation; tasks may be symbolic, vector, or control-oriented.",
        action="Return actions matching `action_space`; action semantics are defined by the upstream task family.",
        reward="Upstream rewards may be vector-valued or multi-component. EvoPolicyGym reports scalar returns when exposed by the adapter.",
        notes=("Multi-objective scalarization policy is not yet calibrated for L3 scoring.",),
    ),
    "BrowserGym MiniWoB++": _Template(
        objective="Complete the browser interaction task `{env_id}` by selecting, clicking, or entering text as required.",
        observation="May include DOM-like fields, screenshots, text instructions, or browser state dictionaries.",
        action="Return the browser action object or discrete command matching `action_space`.",
        reward="Rewards usually indicate task completion success, with sparse feedback for failed interactions.",
        notes=("Browser tasks may require additional runtime services before they can move beyond bulk smoke level.",),
    ),
}


__all__ = ["task_doc"]
