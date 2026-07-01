const data = window.SHOWCASE_DATA;

init();

function init() {
  document.querySelector("#stat-models").textContent = data.models.length;
  document.querySelector("#stat-envs").textContent = data.envs.length;
  document.querySelector("#stat-clips").textContent = data.clips.length;
  document.querySelector("#generated-at").textContent =
    `Generated ${formatDate(data.generated_at)}. ${data.clip_policy}.`;

  renderEnvironmentNav();
  renderEnvironmentSections();
  renderMatrix();
}

function renderEnvironmentNav() {
  const root = document.querySelector("#env-nav");
  if (!root) return;

  const categoryCounts = new Map();
  data.envs.forEach((env) => {
    categoryCounts.set(env.category, (categoryCounts.get(env.category) || 0) + 1);
  });

  root.innerHTML = `
    <a class="env-nav-all" href="#envs">All ${data.envs.length} environments</a>
    ${[...categoryCounts]
      .map(
        ([category, count]) => `
          <a href="#cat-${slugify(category)}">
            <span>${escapeHtml(category)}</span>
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
          <p class="eyebrow">${escapeHtml(env.category)}</p>
          <h2>${escapeHtml(env.display)}</h2>
          <p>${escapeHtml(envSummary(env))}</p>
        </div>
        <div class="env-metrics" aria-label="${escapeHtml(env.display)} result summary">
          <div class="env-winner">
            <span>Best final score</span>
            <strong>${escapeHtml(winner?.model_display || "-")}</strong>
            <em>${formatScore(winner?.score)}</em>
          </div>
          <div>
            <span>Score range</span>
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
      <img src="${clip.media}" alt="${escapeHtml(clip.model_display)} final rollout on ${escapeHtml(clip.env_display)}" loading="lazy">
    </div>
    <div class="card-body">
      <div class="card-title">
        <div>
          <h3>${escapeHtml(clip.model_display)}</h3>
          <span>${escapeHtml(clip.harness || "")}</span>
        </div>
        <span class="rank-pill">${isWinner ? "Best" : `Rank ${rank || "-"}`}</span>
      </div>
      <dl class="card-stats">
        <div>
          <dt>Final score</dt>
          <dd>${formatScore(result.score)}</dd>
        </div>
        <div>
          <dt>Checkpoint</dt>
          <dd>${pad(clip.submit_index)}</dd>
        </div>
        <div>
          <dt>Rerun steps</dt>
          <dd>${formatInteger(clip.rerun_steps)}</dd>
        </div>
      </dl>
      <p>${escapeHtml(clip.event_notes)}</p>
      <div class="artifact-line">
        <span>${isRerun ? captureLabel(clip.capture_source) : "Run artifact"}</span>
        <code>${escapeHtml(isRerun ? rerunLabel(clip) : shortPath(clip.trajectory_path))}</code>
      </div>
    </div>
  `;
  return card;
}

function envSummary(env) {
  const summaries = {
    Control: "Low-dimensional control tasks that test whether agents can discover compact feedback policies.",
    Box2D: "Physics tasks spanning state control and pixel-based driving behavior.",
    MuJoCo: "Continuous-control locomotion and manipulation tasks with dense state vectors.",
    MiniGrid: "Symbolic navigation and planning tasks where policy structure and memory matter.",
    Driving: "HighwayEnv traffic tasks with structured vehicle state and scenario-specific control.",
    Robotics: "Goal-conditioned manipulation tasks from Gymnasium-Robotics.",
  };
  return summaries[env.category] || "A Core16 task exposed through the same EvoPolicyGym policy interface.";
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
      <th>Model</th>
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
    bits.push(`held-out case ${pad(clip.rerun_case_index)}`);
  }
  if (clip.rerun_steps !== null && clip.rerun_steps !== undefined) {
    bits.push(`${clip.rerun_steps} steps`);
  }
  return bits.join(" / ");
}

function captureLabel(source) {
  if (source === "direct_mujoco_renderer") return "MuJoCo native render";
  if (source === "highway_state_renderer") return "Highway scene render";
  if (source === "native_env_render") return "Native env render";
  if (source === "state_capture") return "State capture";
  return "Original env capture";
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
  return date.toLocaleString(undefined, {
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
