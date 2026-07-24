export interface EnvironmentMeta {
  id: string;
  shortEn: string;
  shortZh: string;
  focusEn: string;
  focusZh: string;
}

export const environmentMeta: Record<string, EnvironmentMeta> = {
  acrobot: {
    id: "acrobot",
    shortEn: "Swing-up control with a two-link underactuated arm.",
    shortZh: "双连杆欠驱动机械臂的摆起控制任务。",
    focusEn: "Momentum building under sparse success feedback",
    focusZh: "在稀疏成功反馈下积累动量",
  },
  continuouscar: {
    id: "continuouscar",
    shortEn: "Continuous throttle control with delayed goal reward.",
    shortZh: "带延迟目标奖励的连续油门控制任务。",
    focusEn: "Energy-efficient acceleration and hill climbing",
    focusZh: "兼顾能耗的加速与爬坡",
  },
  bipedal: {
    id: "bipedal",
    shortEn: "Contact-rich locomotion across uneven terrain.",
    shortZh: "在不平地形上的高接触双足运动任务。",
    focusEn: "Balance, gait formation, and forward progress",
    focusZh: "平衡、步态形成与前进效率",
  },
  racing: {
    id: "racing",
    shortEn: "Pixel-observation driving on procedurally generated tracks.",
    shortZh: "在程序生成赛道上的像素观测驾驶任务。",
    focusEn: "Road perception, steering, and recovery",
    focusZh: "道路感知、转向与失控恢复",
  },
  reacher5: {
    id: "reacher5",
    shortEn: "Continuous robotic-arm control toward a target point.",
    shortZh: "控制连续动作机械臂到达目标点。",
    focusEn: "Accurate and efficient end-effector positioning",
    focusZh: "末端执行器的准确高效定位",
  },
  halfcheetah5: {
    id: "halfcheetah5",
    shortEn: "High-dimensional planar locomotion with dense feedback.",
    shortZh: "带稠密反馈的高维平面运动任务。",
    focusEn: "Coordinated speed, stability, and energy use",
    focusZh: "速度、稳定性与能耗的协调",
  },
  ant5: {
    id: "ant5",
    shortEn: "Multi-legged locomotion with eight continuous actions.",
    shortZh: "包含八维连续动作的多足运动任务。",
    focusEn: "Whole-body coordination without collapse",
    focusZh: "避免失衡的全身协调",
  },
  pusher5: {
    id: "pusher5",
    shortEn: "Contact manipulation that pushes an object to a goal.",
    shortZh: "通过接触操作把物体推向目标位置。",
    focusEn: "Arm positioning and object-contact dynamics",
    focusZh: "机械臂定位与物体接触动力学",
  },
  minigrid_doorkey: {
    id: "minigrid_doorkey",
    shortEn: "Find a key, unlock a door, and reach the goal.",
    shortZh: "寻找钥匙、打开门并到达目标。",
    focusEn: "Sparse-reward planning and interaction order",
    focusZh: "稀疏奖励下的规划与交互顺序",
  },
  minigrid_keycorridor: {
    id: "minigrid_keycorridor",
    shortEn: "Multi-room search with object interaction and memory.",
    shortZh: "需要物体交互与记忆的多房间搜索。",
    focusEn: "Exploration, key retrieval, and long-horizon state",
    focusZh: "探索、取钥匙与长时程状态",
  },
  minigrid_fourrooms: {
    id: "minigrid_fourrooms",
    shortEn: "Partial-observation navigation through bottlenecks.",
    shortZh: "在部分观测下穿越四房间瓶颈。",
    focusEn: "Spatial memory and reliable corridor traversal",
    focusZh: "空间记忆与稳定穿越走廊",
  },
  minigrid_obstructedmaze: {
    id: "minigrid_obstructedmaze",
    shortEn: "Maze navigation with blockers and interaction sequences.",
    shortZh: "带阻挡物与交互顺序的迷宫导航。",
    focusEn: "Obstacle handling without losing task progress",
    focusZh: "处理障碍同时保持任务进度",
  },
  parking: {
    id: "parking",
    shortEn: "Continuous vehicle control toward a target parking pose.",
    shortZh: "把车辆连续控制到目标停车姿态。",
    focusEn: "Precise maneuvering and collision avoidance",
    focusZh: "精确机动与碰撞规避",
  },
  roundabout: {
    id: "roundabout",
    shortEn: "Structured traffic control through a roundabout.",
    shortZh: "在环岛场景中的结构化交通控制。",
    focusEn: "Progress, lane choice, and traffic awareness",
    focusZh: "通行效率、车道选择与交通感知",
  },
  fetch_push: {
    id: "fetch_push",
    shortEn: "Goal-conditioned robotic pushing with a gripper.",
    shortZh: "使用机械臂夹爪完成目标条件推动。",
    focusEn: "Object-goal alignment through stable contact",
    focusZh: "通过稳定接触实现物体与目标对齐",
  },
  fetch_pickandplace: {
    id: "fetch_pickandplace",
    shortEn: "Grasp, lift, and place an object at a target.",
    shortZh: "抓取、抬起并把物体放到目标位置。",
    focusEn: "Sequenced grasping and three-dimensional placement",
    focusZh: "按序抓取与三维放置",
  },
};
