const data = window.SHOWCASE_DATA;
const LANGUAGES = ["en", "zh"];

const TEXT = {
  en: {
    "nav.aria": "Showcase navigation",
    "nav.protocol": "Protocol",
    "nav.rollouts": "Rollouts",
    "nav.scores": "Scores",
    "nav.essay": "Essay",
    "language.aria": "Language",
    "hero.eyebrow": "Paper companion",
    "hero.kicker":
      "A benchmark for studying whether coding agents can iteratively improve executable policies through environment feedback.",
    "hero.lede":
      "This companion page collects the final validation-selected Core16 policies, their original-environment reruns, and the held-out scores reported in the paper.",
    "hero.primaryAction": "View Rollouts",
    "hero.secondaryAction": "Score Matrix",
    "hero.essayAction": "Protocol Essay",
    "hero.resources.aria": "Project links",
    "hero.link.github": "GitHub",
    "hero.link.hfPaper": "Hugging Face Paper",
    "hero.link.paper": "arXiv Paper",
    "stats.models": "harness-model agents",
    "stats.envs": "Gymnasium-style tasks",
    "stats.clips": "final policy reruns",
    "protocol.aria": "Benchmark protocol",
    "protocol.eyebrow": "Benchmark protocol",
    "protocol.title": "Agents write policies, not action traces.",
    "protocol.copy":
      "A run starts with a task contract, a policy workspace, and a fixed episode budget. The agent reads visible train feedback, edits the executable policy system, submits checkpoints, and continues from its own accepted workspace state.",
    "protocol.essayLink": "Read the protocol design essay",
    "protocol.facts.online.title": "Online loop",
    "protocol.facts.online.copy": "edit code, submit, inspect visible train feedback",
    "protocol.facts.selection.title": "Selection",
    "protocol.facts.selection.copy": "checkpoint chosen by hidden validation",
    "protocol.facts.measurement.title": "Measurement",
    "protocol.facts.measurement.copy": "final score is hidden held-out mean return",
    "protocol.facts.evidence.title": "Evidence",
    "protocol.facts.evidence.copy":
      "videos rerun validation-selected checkpoints in the original environment",
    "intro.aria": "What is shown",
    "intro.eyebrow": "Final policy gallery",
    "intro.title": "One environment at a time, four agents side by side.",
    "intro.copyA":
      "The gallery is organized by environment. Each row compares the final selected checkpoint from GPT-5.5 through Codex, and Claude Opus 4.7, MiniMax-M3, and DeepSeek-V4-Pro through the Claude Code-compatible harness.",
    "intro.copyB":
      "The clips are intentionally slower than the raw rollout cadence so qualitative behavior is readable: locomotion, traffic, manipulation, navigation, and failure modes remain visible without showing the intermediate optimization process.",
    "envNav.aria": "Environment quick links",
    "envSections.aria": "Environment final policy sections",
    "matrix.eyebrow": "Reported scores",
    "matrix.title": "Validation-selected held-out return.",
    "matrix.copy":
      "Scores are raw environment returns from the selected final checkpoint. Higher is better within each environment; reward scales are not comparable across columns.",
    "footer.title": "EvoPolicyGym companion gallery",
    "footer.about":
      "An interactive companion for the Core16 benchmark, pairing validation-selected policy reruns with held-out scores so readers can compare how coding agents behave across tasks.",
    allEnvs: (count) => `All ${count} environments`,
    generated: (date, policy) => `Generated ${date}. ${policy}.`,
    clipPolicyFinal: "one clip per model-env lane from the run's validation-selected checkpoint",
    bestFinalScore: "Best final score",
    scoreRange: "Score range",
    scoreNote: "Higher scores indicate better performance within this task.",
    rankBest: "Best",
    rank: (rank) => `Rank ${rank || "-"}`,
    finalScore: "Final score",
    checkpoint: "Checkpoint",
    rerunSteps: "Rerun steps",
    model: "Model",
    heldoutCase: (index) => `held-out case ${pad(index)}`,
    steps: (steps) => `${steps} steps`,
    runArtifact: "Run artifact",
    capture: {
      direct_mujoco_renderer: "MuJoCo native render",
      highway_state_renderer: "Highway scene render",
      native_env_render: "Native env render",
      state_capture: "State capture",
      fallback: "Original env capture",
    },
    finalEventPrefix: "Validation-selected checkpoint rerun in the original environment",
    category: {
      Control: "Control",
      Box2D: "Box2D",
      MuJoCo: "MuJoCo",
      MiniGrid: "MiniGrid",
      Driving: "Driving",
      Robotics: "Robotics",
    },
    categorySummary: {
      Control:
        "Low-dimensional control tasks that test whether agents can discover compact feedback policies.",
      Box2D: "Physics tasks spanning state control and pixel-based driving behavior.",
      MuJoCo: "Continuous-control locomotion and manipulation tasks with dense state vectors.",
      MiniGrid: "Symbolic navigation and planning tasks where policy structure and memory matter.",
      Driving: "HighwayEnv traffic tasks with structured vehicle state and scenario-specific control.",
      Robotics: "Goal-conditioned manipulation tasks from Gymnasium-Robotics.",
      fallback: "A Core16 task exposed through the same EvoPolicyGym policy interface.",
    },
    taskSummary: {
      acrobot:
        "Swing-up control task with a two-link underactuated arm. The policy must build momentum and raise the end effector above a target height under sparse success feedback.",
      continuouscar:
        "Continuous throttle control task with delayed goal reward. The policy must trade off acceleration cost against building enough momentum to reach the hilltop goal.",
      bipedal:
        "Contact-rich locomotion task. The policy must keep a two-legged walker balanced while moving forward across uneven terrain.",
      racing:
        "Pixel-observation driving task. The policy must infer the road from rendered frames, steer through the track, and avoid leaving the course.",
      reacher5:
        "Continuous-control manipulation task with dense state vectors. The goal is to control a simulated robotic arm so its end effector reaches a target location accurately and quickly.",
      halfcheetah5:
        "High-dimensional locomotion task. The policy must coordinate a planar cheetah body to run forward quickly while controlling energy and stability.",
      ant5:
        "Multi-legged locomotion task. The policy must coordinate eight continuous actions to move a quadruped forward without losing balance.",
      pusher5:
        "Robotic manipulation task with contact dynamics. The policy must move the arm to push an object toward a goal location.",
      minigrid_doorkey:
        "Symbolic planning task with sparse reward. The policy must find a key, unlock the door, and navigate to the goal.",
      minigrid_keycorridor:
        "Multi-room symbolic navigation task. The policy must search rooms, retrieve the right key, and use object interactions to reach the goal.",
      minigrid_fourrooms:
        "Sparse-reward navigation task. The policy must move through four connected rooms and bottlenecks under partial observations.",
      minigrid_obstructedmaze:
        "Maze-navigation task with obstacles and interaction sequencing. The policy must clear or route around blockers while preserving progress to the goal.",
      parking:
        "Goal-directed driving task with continuous vehicle control. The policy must maneuver into a target parking pose precisely while avoiding collisions.",
      roundabout:
        "Structured traffic task. The policy must enter and navigate a roundabout while balancing progress, lane choice, and collision avoidance.",
      fetch_push:
        "Goal-conditioned robotics task. The policy must use the robot gripper to push a block so the achieved object position matches the target goal.",
      fetch_pickandplace:
        "Goal-conditioned manipulation task. The policy must grasp, lift, and place an object at the desired target position.",
    },
  },
  zh: {
    "nav.aria": "展示页导航",
    "nav.protocol": "协议",
    "nav.rollouts": "回放",
    "nav.scores": "分数",
    "nav.essay": "解读",
    "language.aria": "语言",
    "hero.eyebrow": "论文配套页面",
    "hero.kicker": "一个用于研究 coding agent 是否能通过环境反馈迭代改进可执行策略的 benchmark。",
    "hero.lede":
      "本页面汇总 Core16 中按验证集选择的最终策略、这些策略在原始环境中的重新运行视频，以及论文报告的 held-out 分数。",
    "hero.primaryAction": "查看回放",
    "hero.secondaryAction": "分数矩阵",
    "hero.essayAction": "协议解读",
    "hero.resources.aria": "项目链接",
    "hero.link.github": "GitHub",
    "hero.link.hfPaper": "Hugging Face Paper",
    "hero.link.paper": "arXiv 论文",
    "stats.models": "harness-model agents",
    "stats.envs": "Gymnasium 风格任务",
    "stats.clips": "最终策略回放",
    "protocol.aria": "Benchmark 协议",
    "protocol.eyebrow": "Benchmark 协议",
    "protocol.title": "Agents 编写策略，而不是动作轨迹。",
    "protocol.copy":
      "每次运行从任务契约、策略工作区和固定 episode budget 开始。Agent 读取可见训练反馈，编辑可执行策略系统，提交 checkpoint，并从自己已接受的工作区状态继续优化。",
    "protocol.essayLink": "阅读协议设计解读",
    "protocol.facts.online.title": "在线循环",
    "protocol.facts.online.copy": "改代码、提交、查看可见训练反馈",
    "protocol.facts.selection.title": "选择",
    "protocol.facts.selection.copy": "checkpoint 由隐藏验证集选择",
    "protocol.facts.measurement.title": "测量",
    "protocol.facts.measurement.copy": "最终分数是隐藏 held-out mean return",
    "protocol.facts.evidence.title": "视频证据",
    "protocol.facts.evidence.copy": "视频是在原始环境中重新运行 validation-selected checkpoint",
    "intro.aria": "页面展示内容",
    "intro.eyebrow": "最终策略展示",
    "intro.title": "按环境展示，四个 agent 并排比较。",
    "intro.copyA":
      "展示按环境组织。每一行比较 GPT-5.5 通过 Codex 得到的最终 selected checkpoint，以及 Claude Opus 4.7、MiniMax-M3、DeepSeek-V4-Pro 通过 Claude Code-compatible harness 得到的对应 checkpoint。",
    "intro.copyB":
      "视频播放速度有意低于原始 rollout 速度，便于观察定性行为：移动、交通、机械臂操作、导航和失败模式都可以直接比较，但不展示中间优化过程。",
    "envNav.aria": "环境快速导航",
    "envSections.aria": "环境最终策略区块",
    "matrix.eyebrow": "论文报告分数",
    "matrix.title": "Validation-selected held-out return.",
    "matrix.copy": "分数是 selected checkpoint 的原始环境 return。同一环境内分数越高越好；不同列的奖励尺度不可直接比较。",
    "footer.title": "EvoPolicyGym 配套展示页",
    "footer.about":
      "一个面向 Core16 benchmark 的交互式展示页，将 validation-selected 策略回放与 held-out 分数并排呈现，帮助读者按任务比较 coding agents 的行为差异。",
    allEnvs: (count) => `全部 ${count} 个环境`,
    generated: (date, policy) => `生成时间 ${date}。${policy}。`,
    clipPolicyFinal: "每个 model-env lane 展示一个来自 validation-selected checkpoint 的视频",
    bestFinalScore: "最佳最终分数",
    scoreRange: "分数范围",
    scoreNote: "在该任务内，分数越高表示表现越好。",
    rankBest: "最佳",
    rank: (rank) => `第 ${rank || "-"} 名`,
    finalScore: "最终分数",
    checkpoint: "Checkpoint",
    rerunSteps: "回放步数",
    model: "模型",
    heldoutCase: (index) => `held-out case ${pad(index)}`,
    steps: (steps) => `${steps} 步`,
    runArtifact: "运行产物",
    capture: {
      direct_mujoco_renderer: "MuJoCo 原生渲染",
      highway_state_renderer: "Highway 场景渲染",
      native_env_render: "环境原生渲染",
      state_capture: "状态捕获",
      fallback: "原始环境捕获",
    },
    finalEventPrefix: "在原始环境中重新运行 validation-selected checkpoint",
    category: {
      Control: "控制",
      Box2D: "Box2D",
      MuJoCo: "MuJoCo",
      MiniGrid: "MiniGrid",
      Driving: "驾驶",
      Robotics: "机器人",
    },
    categorySummary: {
      Control: "低维控制任务，用于测试 agent 是否能发现紧凑的反馈策略。",
      Box2D: "物理任务，覆盖状态控制和像素驾驶行为。",
      MuJoCo: "基于稠密状态向量的连续控制移动和操作任务。",
      MiniGrid: "符号导航与规划任务，策略结构和记忆很重要。",
      Driving: "HighwayEnv 交通任务，具有结构化车辆状态和场景化控制目标。",
      Robotics: "来自 Gymnasium-Robotics 的 goal-conditioned 操作任务。",
      fallback: "一个通过相同 EvoPolicyGym 策略接口暴露的 Core16 任务。",
    },
    taskSummary: {
      acrobot: "双连杆欠驱动摆起控制任务。策略需要积累动量，让末端超过目标高度；成功反馈较稀疏。",
      continuouscar: "连续油门控制任务，目标奖励延迟出现。策略需要在动作代价和积累足够动量到达山顶之间权衡。",
      bipedal: "接触丰富的双足行走任务。策略需要保持 walker 平衡，并在不平坦地形上向前移动。",
      racing: "像素观测驾驶任务。策略需要从渲染图像中识别道路，沿赛道转向，并避免驶离路线。",
      reacher5:
        "基于稠密状态向量的连续控制机械臂任务。目标是控制模拟机械臂，使末端执行器尽可能准确且快速地到达目标位置。",
      halfcheetah5: "高维连续运动任务。策略需要协调平面 cheetah 身体快速前进，同时控制能量消耗和稳定性。",
      ant5: "多足运动任务。策略需要协调 8 个连续动作，让四足体向前移动并保持平衡。",
      pusher5: "带接触动力学的机器人操作任务。策略需要移动机械臂，将物体推向目标位置。",
      minigrid_doorkey: "稀疏奖励符号规划任务。策略需要找到钥匙、开门，并导航到目标位置。",
      minigrid_keycorridor: "多房间符号导航任务。策略需要搜索房间，取到正确钥匙，并通过物体交互到达目标。",
      minigrid_fourrooms: "稀疏奖励导航任务。策略需要在部分观测下穿过四个相连房间和瓶颈通道。",
      minigrid_obstructedmaze: "带障碍和交互顺序要求的迷宫导航任务。策略需要清除或绕过阻挡，同时保持向目标推进。",
      parking: "连续车辆控制的目标导向驾驶任务。策略需要精准停入目标姿态并避免碰撞。",
      roundabout: "结构化交通任务。策略需要进入并通过环岛，在推进、车道选择和避免碰撞之间权衡。",
      fetch_push: "Goal-conditioned 机器人任务。策略需要用夹爪推动方块，使 achieved object position 匹配目标。",
      fetch_pickandplace: "Goal-conditioned 操作任务。策略需要抓取、抬起并把物体放到目标位置。",
    },
  },
};

let currentLang = initialLanguage();

init();

function init() {
  document.querySelector("#stat-models").textContent = data.models.length;
  document.querySelector("#stat-envs").textContent = data.envs.length;
  document.querySelector("#stat-clips").textContent = data.clips.length;

  setupLanguageToggle();
  renderPage();
}

function renderPage() {
  document.documentElement.lang = currentLang === "zh" ? "zh-CN" : "en";
  applyStaticTranslations();
  updateLanguageToggle();
  renderEnvironmentNav();
  renderEnvironmentSections();
  renderMatrix();
}

function setupLanguageToggle() {
  document.querySelectorAll("[data-lang-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const nextLang = button.getAttribute("data-lang-toggle");
      if (!LANGUAGES.includes(nextLang) || nextLang === currentLang) return;
      currentLang = nextLang;
      try {
        localStorage.setItem("showcase-language", currentLang);
      } catch {
        // Language preference is optional; rendering does not depend on storage.
      }
      renderPage();
    });
  });
}

function applyStaticTranslations() {
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.getAttribute("data-i18n"));
  });
  document.querySelectorAll("[data-i18n-html]").forEach((node) => {
    node.innerHTML = t(node.getAttribute("data-i18n-html"));
  });
  document.querySelectorAll("[data-i18n-aria]").forEach((node) => {
    node.setAttribute("aria-label", t(node.getAttribute("data-i18n-aria")));
  });
}

function updateLanguageToggle() {
  document.querySelectorAll("[data-lang-toggle]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.getAttribute("data-lang-toggle") === currentLang));
  });
}

function renderEnvironmentNav() {
  const root = document.querySelector("#env-nav");
  if (!root) return;

  const categoryCounts = new Map();
  data.envs.forEach((env) => {
    categoryCounts.set(env.category, (categoryCounts.get(env.category) || 0) + 1);
  });

  root.innerHTML = `
    <a class="env-nav-all" href="#envs">${escapeHtml(text().allEnvs(data.envs.length))}</a>
    ${[...categoryCounts]
      .map(
        ([category, count]) => `
          <a href="#cat-${slugify(category)}">
            <span>${escapeHtml(categoryLabel(category))}</span>
            <em>${count}</em>
          </a>
        `,
      )
      .join("")}
  `;
}

function renderEnvironmentSections() {
  const root = document.querySelector("#envs");
  const byLane = new Map(data.clips.map((clip) => [`${clip.env_id}|${clip.model_slug}`, clip]));
  const results = new Map(data.final_results.map((row) => [`${row.env_id}|${row.model_slug}`, row]));
  const seenCategories = new Set();

  root.innerHTML = "";
  data.envs.forEach((env) => {
    const clips = data.models
      .map((model) => byLane.get(`${env.id}|${model.slug}`))
      .filter(Boolean);
    if (!clips.length) return;

    const winner = bestResult(env.id);
    const spread = scoreSpread(env.id);
    const categoryAnchor = seenCategories.has(env.category)
      ? ""
      : ` id="cat-${slugify(env.category)}"`;
    seenCategories.add(env.category);

    const section = document.createElement("section");
    section.className = "env-section";
    section.id = `env-${env.id}`;
    section.innerHTML = `
      <div class="env-heading"${categoryAnchor}>
        <div>
          <p class="eyebrow">${escapeHtml(categoryLabel(env.category))}</p>
          <h2>${escapeHtml(env.display)}</h2>
          <p>${escapeHtml(envSummary(env))}</p>
          <p class="score-note">${escapeHtml(text().scoreNote)}</p>
        </div>
        <div class="env-metrics" aria-label="${escapeHtml(env.display)} result summary">
          <div class="env-winner">
            <span>${escapeHtml(text().bestFinalScore)}</span>
            <strong>${escapeHtml(winner?.model_display || "-")}</strong>
            <em>${formatScore(winner?.score)}</em>
          </div>
          <div>
            <span>${escapeHtml(text().scoreRange)}</span>
            <strong>${formatScore(spread?.low)} - ${formatScore(spread?.high)}</strong>
            <em>${formatScore(spread?.delta)}</em>
          </div>
        </div>
      </div>
      <div class="model-grid"></div>
    `;

    const grid = section.querySelector(".model-grid");
    data.models.forEach((model) => {
      const clip = byLane.get(`${env.id}|${model.slug}`);
      const result = results.get(`${env.id}|${model.slug}`);
      if (!clip || !result) return;
      grid.append(renderModelCard(clip, result, winner));
    });
    root.append(section);
  });
}

function renderModelCard(clip, result, winner) {
  const isWinner = winner && winner.model_slug === clip.model_slug;
  const isRerun = clip.media_source === "original_env_rerun";
  const rank = envRank(clip.env_id, clip.model_slug);
  const card = document.createElement("article");
  card.className = `model-card${isWinner ? " winner" : ""}`;
  card.innerHTML = `
    <div class="media-shell">
      <img src="${clip.media}" alt="${escapeHtml(clipAlt(clip))}" loading="lazy">
    </div>
    <div class="card-body">
      <div class="card-title">
        <div>
          <h3>${escapeHtml(clip.model_display)}</h3>
          <span>${escapeHtml(clip.harness || "")}</span>
        </div>
        <span class="rank-pill">${escapeHtml(isWinner ? text().rankBest : text().rank(rank))}</span>
      </div>
      <dl class="card-stats">
        <div>
          <dt>${escapeHtml(text().finalScore)}</dt>
          <dd>${formatScore(result.score)}</dd>
        </div>
        <div>
          <dt>${escapeHtml(text().checkpoint)}</dt>
          <dd>${pad(clip.submit_index)}</dd>
        </div>
        <div>
          <dt>${escapeHtml(text().rerunSteps)}</dt>
          <dd>${formatInteger(clip.rerun_steps)}</dd>
        </div>
      </dl>
      <p>${escapeHtml(eventNotes(clip))}</p>
      <div class="artifact-line">
        <span>${escapeHtml(isRerun ? captureLabel(clip.capture_source) : text().runArtifact)}</span>
        <code>${escapeHtml(isRerun ? rerunLabel(clip) : shortPath(clip.trajectory_path))}</code>
      </div>
    </div>
  `;
  return card;
}

function envSummary(env) {
  return text().taskSummary[env.id] || text().categorySummary[env.category] || text().categorySummary.fallback;
}

function categoryLabel(category) {
  return text().category[category] || category;
}

function eventNotes(clip) {
  if (currentLang === "zh" && typeof clip.event_notes === "string") {
    const match = clip.event_notes.match(/case return (.+)\.$/);
    if (match) return `${text().finalEventPrefix}；case return ${match[1]}。`;
  }
  return clip.event_notes;
}

function clipAlt(clip) {
  if (currentLang === "zh") {
    return `${clip.model_display} 在 ${clip.env_display} 上的最终策略回放`;
  }
  return `${clip.model_display} final rollout on ${clip.env_display}`;
}

function bestResult(envId) {
  const rows = data.final_results
    .filter((row) => row.env_id === envId && row.score !== null)
    .map((row) => ({
      ...row,
      model_display: data.models.find((model) => model.slug === row.model_slug)?.display || row.model_slug,
    }));
  if (!rows.length) return null;
  return rows.reduce((best, row) => (row.score > best.score ? row : best), rows[0]);
}

function scoreSpread(envId) {
  const scores = data.final_results
    .filter((row) => row.env_id === envId && row.score !== null && row.score !== undefined)
    .map((row) => Number(row.score));
  if (!scores.length) return null;
  const low = Math.min(...scores);
  const high = Math.max(...scores);
  return { low, high, delta: high - low };
}

function envRank(envId, modelSlug) {
  const rows = data.final_results
    .filter((row) => row.env_id === envId && row.score !== null && row.score !== undefined)
    .sort((a, b) => Number(b.score) - Number(a.score));
  const index = rows.findIndex((row) => row.model_slug === modelSlug);
  return index >= 0 ? index + 1 : null;
}

function renderMatrix() {
  const table = document.querySelector("#score-matrix");
  const byLane = new Map(data.final_results.map((row) => [`${row.model_slug}|${row.env_id}`, row]));
  const bestByEnv = new Map();
  data.envs.forEach((env) => {
    const rows = data.final_results.filter((row) => row.env_id === env.id && row.score !== null);
    if (rows.length) bestByEnv.set(env.id, Math.max(...rows.map((row) => row.score)));
  });

  table.innerHTML = "";
  const thead = document.createElement("thead");
  thead.innerHTML = `
    <tr>
      <th>${escapeHtml(text().model)}</th>
      ${data.envs.map((env) => `<th>${escapeHtml(shortEnv(env.display))}</th>`).join("")}
    </tr>
  `;
  table.append(thead);

  const tbody = document.createElement("tbody");
  data.models.forEach((model) => {
    const tr = document.createElement("tr");
    const cells = data.envs
      .map((env) => {
        const result = byLane.get(`${model.slug}|${env.id}`);
        const score = result?.score;
        const best = bestByEnv.get(env.id);
        const cls = score !== null && score !== undefined && best === score ? "best" : "";
        return `<td class="${cls}">${formatScore(score)}</td>`;
      })
      .join("");
    tr.innerHTML = `<td>${escapeHtml(model.display)}</td>${cells}`;
    tbody.append(tr);
  });
  table.append(tbody);
}

function shortEnv(value) {
  return value
    .replace("MiniGrid-", "")
    .replace("MountainCar Continuous", "ContinuousCar")
    .replace("FetchPickAndPlace", "PickPlace");
}

function shortPath(value) {
  if (!value) return "";
  const parts = String(value).split("/");
  const submit = parts.find((part) => part.startsWith("submit_"));
  const episode = parts.find((part) => part.startsWith("ep_"));
  return [submit, episode, "trajectory.jsonl"].filter(Boolean).join(" / ");
}

function rerunLabel(clip) {
  const bits = [];
  if (clip.rerun_case_index !== null && clip.rerun_case_index !== undefined) {
    bits.push(text().heldoutCase(clip.rerun_case_index));
  }
  if (clip.rerun_steps !== null && clip.rerun_steps !== undefined) {
    bits.push(text().steps(clip.rerun_steps));
  }
  return bits.join(" / ");
}

function captureLabel(source) {
  return text().capture[source] || text().capture.fallback;
}

function clipPolicyLabel(value) {
  if (
    value === "one clip per model-env lane from the run's validation-selected checkpoint" &&
    currentLang === "zh"
  ) {
    return text().clipPolicyFinal;
  }
  return value;
}

function text() {
  return TEXT[currentLang] || TEXT.en;
}

function t(key) {
  return text()[key] ?? TEXT.en[key] ?? "";
}

function initialLanguage() {
  const params = new URLSearchParams(window.location.search);
  const queryLang = params.get("lang");
  if (LANGUAGES.includes(queryLang)) return queryLang;
  try {
    const stored = localStorage.getItem("showcase-language");
    if (LANGUAGES.includes(stored)) return stored;
  } catch {
    // Ignore storage access errors.
  }
  const navLang = navigator.language || "";
  return navLang.toLowerCase().startsWith("zh") ? "zh" : "en";
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const abs = Math.abs(Number(value));
  if (abs >= 100) return Number(value).toFixed(1);
  if (abs >= 10) return Number(value).toFixed(2);
  return Number(value).toFixed(3);
}

function formatInteger(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return String(Math.round(Number(value)));
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(currentLang === "zh" ? "zh-CN" : undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function pad(value) {
  return String(value).padStart(3, "0");
}

function slugify(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
