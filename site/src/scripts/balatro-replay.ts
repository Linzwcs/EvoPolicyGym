type Scalar = string | number | boolean | null;
type ReplayValue = Scalar | ReplayValue[] | { [key: string]: ReplayValue };

interface ReplayCard {
  index: number;
  key: string | null;
  name: string;
  set: string | null;
  facing: string;
  rank: string | null;
  suit: string | null;
  chips: number | null;
  edition: string | null;
  seal: string | null;
  debuffed: boolean;
  cost: number;
  sell_value: number;
  ability: Record<string, ReplayValue>;
}

interface ReplayState {
  phase: string;
  progress: {
    ante: number;
    rounds_cleared: number;
    win_ante: number;
    blind_on_deck: string | null;
    won: boolean;
    steps: number;
  };
  resources: {
    money: number;
    chips: number;
    hands_left: number;
    discards_left: number;
    hand_size: number;
    joker_slots: number;
    consumable_slots: number;
  };
  blind: {
    name: string;
    target_chips: number;
    dollar_reward: number;
    boss: boolean;
    skip_tag?: {
      key: string;
      name: string;
      rule: {
        summary: string;
        parameters: Record<string, unknown>;
      };
    } | null;
  } | null;
  last_hand: {
    chips?: number;
    mult?: number;
    handname?: string;
  };
  round_earnings: {
    blind_dollar_reward: number;
    unused_hands_bonus: number;
    unused_discards_bonus: number;
    joker_dollars: number;
    interest: number;
    rental_cost: number;
    total_dollars: number;
  } | null;
  hand: ReplayCard[];
  jokers: ReplayCard[];
  consumables: ReplayCard[];
  shop: {
    cards: ReplayCard[];
    vouchers: ReplayCard[];
    boosters: ReplayCard[];
  };
  pack: {
    type: string;
    choices_remaining: number;
    cards: ReplayCard[];
  };
}

interface ReplayAction {
  kind: string;
  card_indices?: number[];
  target_index?: number;
}

interface ReplayEpisode {
  type: "episode";
  episode_index: number;
  status: string;
  steps: number;
  score: number;
  failure: string | null;
  initial_state: ReplayState;
}

interface ReplayTransition {
  type: "transition";
  episode_index: number;
  step_index: number;
  action: ReplayAction;
  reward: number;
  state: ReplayState;
  terminated: boolean;
  truncated: boolean;
}

interface ReplayFrame {
  state: ReplayState;
  action: ReplayAction | null;
  transition: ReplayTransition | null;
}

const viewer = document.querySelector<HTMLElement>("[data-replay-viewer]");

if (viewer) {
  const required = <T extends Element>(selector: string): T => {
    const node = viewer.querySelector<T>(selector);
    if (!node) throw new Error(`Missing Balatro replay element: ${selector}`);
    return node;
  };

  const playButton = required<HTMLButtonElement>("[data-replay-play]");
  const playSymbol = required<HTMLElement>("[data-play-symbol]");
  const playLabel = required<HTMLElement>("[data-play-label]");
  const previousButton = required<HTMLButtonElement>("[data-replay-prev]");
  const nextButton = required<HTMLButtonElement>("[data-replay-next]");
  const range = required<HTMLInputElement>("[data-replay-range]");
  const speed = required<HTMLSelectElement>("[data-replay-speed]");
  const fileInput = required<HTMLInputElement>("[data-replay-file]");
  const openButton = required<HTMLButtonElement>("[data-replay-open]");
  const resetButton = required<HTMLButtonElement>("[data-replay-reset]");
  const dropTarget = required<HTMLElement>("[data-replay-drop]");
  const eventTrack = required<HTMLElement>("[data-replay-events]");
  const sourceName = required<HTMLElement>("[data-replay-source-name]");
  const status = required<HTMLElement>("[data-replay-status]");

  let episode: ReplayEpisode | null = null;
  let frames: ReplayFrame[] = [];
  let index = 0;
  let playing = false;
  let timer: number | null = null;
  let language: "en" | "zh" = document.documentElement.lang.startsWith("zh") ? "zh" : "en";
  const bundledSource = viewer.dataset.source ?? "";

  const number = new Intl.NumberFormat("en-US");
  const suitSymbols: Record<string, string> = {
    Hearts: "♥",
    Diamonds: "♦",
    Clubs: "♣",
    Spades: "♠",
  };
  const rankSymbols: Record<string, string> = {
    Ace: "A",
    King: "K",
    Queen: "Q",
    Jack: "J",
  };
  const actionCopy: Record<string, [string, string]> = {
    initial_state: ["Initial state", "初始状态"],
    select_blind: ["Enter blind", "进入盲注"],
    skip_blind: ["Skip blind", "跳过盲注"],
    play_hand: ["Play hand", "出牌"],
    discard: ["Discard", "弃牌"],
    cash_out: ["Cash out", "领取奖励"],
    reroll_shop: ["Reroll shop", "刷新商店"],
    next_round: ["Leave shop", "离开商店"],
    buy_card: ["Buy card", "购买卡牌"],
    sell_joker: ["Sell Joker", "出售 Joker"],
    sell_consumable: ["Sell consumable", "出售消耗牌"],
    use_consumable: ["Use consumable", "使用消耗牌"],
    redeem_voucher: ["Redeem voucher", "兑换优惠券"],
    open_booster: ["Open booster", "打开补充包"],
    pick_pack_card: ["Choose pack card", "选择补充包卡牌"],
    skip_pack: ["Skip pack", "跳过补充包"],
  };
  const phaseCopy: Record<string, [string, string]> = {
    blind_select: ["Blind select", "选择盲注"],
    selecting_hand: ["Playing blind", "盲注进行中"],
    round_eval: ["Round complete", "本轮完成"],
    shop: ["Shop", "商店"],
    booster_pack: ["Booster pack", "补充包"],
    game_over: ["Run over", "本局结束"],
  };

  function localized(copy: [string, string] | undefined, fallback: string): string {
    if (!copy) return fallback.replaceAll("_", " ");
    return language === "zh" ? copy[1] : copy[0];
  }

  function setText(selector: string, value: string | number): void {
    required<HTMLElement>(selector).textContent = String(value);
  }

  function parseReplay(text: string): { episode: ReplayEpisode; frames: ReplayFrame[] } {
    const documents = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line, lineIndex) => {
        try {
          return JSON.parse(line) as ReplayEpisode | ReplayTransition | { type: string };
        } catch {
          throw new Error(`Invalid JSON on line ${lineIndex + 1}`);
        }
      });
    const foundEpisode = documents.find((document): document is ReplayEpisode => document.type === "episode");
    if (!foundEpisode?.initial_state) throw new Error("The replay has no episode header");
    const transitions = documents.filter(
      (document): document is ReplayTransition =>
        document.type === "transition" && document.episode_index === foundEpisode.episode_index,
    );
    if (!transitions.length) throw new Error("The replay has no transitions");
    return {
      episode: foundEpisode,
      frames: [
        { state: foundEpisode.initial_state, action: null, transition: null },
        ...transitions.map((transition) => ({
          state: transition.state,
          action: transition.action,
          transition,
        })),
      ],
    };
  }

  function cardElement(card: ReplayCard, compact = false): HTMLElement {
    const playingCard = Boolean(card.rank && card.suit);
    const faceDown = card.facing === "back";
    const element = document.createElement("article");
    element.className = playingCard ? "replay-card is-playing-card" : "replay-card is-entity-card";
    if (compact) element.classList.add("is-compact");
    if (faceDown) element.classList.add("is-face-down");
    if (card.debuffed) element.classList.add("is-debuffed");
    if (card.edition) element.classList.add(`has-edition-${card.edition.toLowerCase()}`);
    element.title = card.name;

    if (playingCard && !faceDown) {
      const suit = suitSymbols[card.suit ?? ""] ?? "?";
      const rank = rankSymbols[card.rank ?? ""] ?? card.rank ?? "?";
      const corner = document.createElement("span");
      corner.className = "replay-card-corner";
      if (card.suit === "Hearts" || card.suit === "Diamonds") corner.classList.add("is-red");
      const rankNode = document.createElement("b");
      rankNode.textContent = rank;
      const suitNode = document.createElement("i");
      suitNode.textContent = suit;
      corner.append(rankNode, suitNode);
      const center = document.createElement("strong");
      center.className = "replay-card-suit";
      center.textContent = suit;
      if (card.suit === "Hearts" || card.suit === "Diamonds") center.classList.add("is-red");
      const chips = document.createElement("small");
      chips.textContent = `${card.chips ?? 0}`;
      element.append(corner, center, chips);
      return element;
    }

    if (faceDown) {
      const pattern = document.createElement("span");
      pattern.className = "replay-card-back";
      pattern.textContent = "EPG";
      element.append(pattern);
      return element;
    }

    const set = document.createElement("span");
    set.className = "replay-entity-set";
    set.textContent = card.set || "Card";
    const name = document.createElement("strong");
    name.textContent = card.name;
    const detail = document.createElement("small");
    const ability = Object.entries(card.ability ?? {}).find(
      ([, value]) => (typeof value === "number" && value !== 0 && value !== 1) || typeof value === "string",
    );
    detail.textContent = ability ? `${ability[0]} ${String(ability[1])}` : card.cost ? `$${card.cost}` : "—";
    element.append(set, name, detail);
    return element;
  }

  function emptyCard(label: string): HTMLElement {
    const element = document.createElement("div");
    element.className = "replay-empty-card";
    element.textContent = label;
    return element;
  }

  function renderCardRow(selector: string, cards: ReplayCard[], slots: number, compact = false): void {
    const container = required<HTMLElement>(selector);
    container.replaceChildren(...cards.map((card) => cardElement(card, compact)));
    const visibleSlots = Math.max(0, Math.min(slots - cards.length, compact ? 2 : 5));
    for (let slot = 0; slot < visibleSlots; slot += 1) {
      container.append(emptyCard("+"));
    }
  }

  function deltaLabel(current: number, previous: number, noun: [string, string]): string {
    const delta = current - previous;
    if (delta === 0) return language === "zh" ? `本步${noun[1]}无变化` : `No ${noun[0]} change`;
    return `${delta > 0 ? "+" : ""}${number.format(delta)} ${language === "zh" ? noun[1] : noun[0]}`;
  }

  function actionDetail(action: ReplayAction | null, frame: ReplayFrame): string {
    if (!action) {
      return language === "zh" ? "等待第一次决策" : "Waiting for the first decision";
    }
    if (action.card_indices?.length) {
      const indices = action.card_indices.map((value) => `#${value + 1}`).join(", ");
      return language === "zh" ? `卡牌 ${indices}` : `Cards ${indices}`;
    }
    if (typeof action.target_index === "number") {
      return language === "zh"
        ? `目标 #${action.target_index + 1}`
        : `Target #${action.target_index + 1}`;
    }
    if (frame.transition?.terminated) {
      return episode?.status === "completed"
        ? language === "zh"
          ? "Episode 已完成"
          : "Episode completed"
        : language === "zh"
          ? "Episode 已终止"
          : "Episode terminated";
    }
    return language === "zh" ? "状态已更新" : "State updated";
  }

  function renderMarket(state: ReplayState): void {
    const market = required<HTMLElement>("[data-replay-market]");
    const inPack = state.pack.cards.length > 0;
    const cards = inPack
      ? state.pack.cards
      : [...state.shop.cards, ...state.shop.vouchers, ...state.shop.boosters];
    setText("[data-replay-market-label]", inPack ? (language === "zh" ? "补充包" : "Booster pack") : language === "zh" ? "商店" : "Shop");
    setText(
      "[data-replay-market-count]",
      language === "zh" ? `${cards.length} 项` : `${cards.length} item${cards.length === 1 ? "" : "s"}`,
    );
    market.replaceChildren();
    if (!cards.length) {
      const empty = document.createElement("p");
      empty.className = "replay-market-empty";
      empty.textContent = language === "zh" ? "当前阶段没有可见商品" : "No visible items in this phase";
      market.append(empty);
      return;
    }
    cards.slice(0, 6).forEach((card) => {
      const row = document.createElement("div");
      const name = document.createElement("strong");
      name.textContent = card.name;
      const meta = document.createElement("span");
      meta.textContent = card.cost > 0 ? `$${card.cost}` : card.set || "Card";
      row.append(name, meta);
      market.append(row);
    });
  }

  function renderEvents(): void {
    eventTrack.replaceChildren();
    frames.forEach((frame, frameIndex) => {
      const event = document.createElement("button");
      event.type = "button";
      event.dataset.frame = String(frameIndex);
      event.className = `replay-event is-${frame.action?.kind ?? "initial"}`;
      event.setAttribute("aria-label", `${frameIndex}: ${localized(actionCopy[frame.action?.kind ?? "initial_state"], "Initial state")}`);
      event.title = localized(actionCopy[frame.action?.kind ?? "initial_state"], "Initial state");
      event.addEventListener("click", () => {
        pause();
        index = frameIndex;
        render();
      });
      eventTrack.append(event);
    });
  }

  function render(): void {
    if (!episode || !frames.length) return;
    const frame = frames[index];
    const state = frame.state;
    const previous = frames[Math.max(0, index - 1)].state;
    const blind = state.blind;

    range.max = String(frames.length - 1);
    range.value = String(index);
    setText("[data-replay-step]", `${String(index).padStart(2, "0")} / ${String(frames.length - 1).padStart(2, "0")}`);
    setText("[data-replay-progress]", `Ante ${state.progress.ante} / ${state.progress.win_ante}`);
    setText(
      "[data-replay-rounds]",
      language === "zh"
        ? `已通过 ${state.progress.rounds_cleared} 个盲注`
        : `${state.progress.rounds_cleared} blind${state.progress.rounds_cleared === 1 ? "" : "s"} cleared`,
    );
    setText("[data-replay-chips]", number.format(state.resources.chips));
    setText("[data-replay-target]", number.format(blind?.target_chips ?? 0));
    setText(
      "[data-replay-chip-delta]",
      deltaLabel(state.resources.chips, previous.resources.chips, ["score", "分数"]),
    );
    setText("[data-replay-money]", state.resources.money);
    setText(
      "[data-replay-money-delta]",
      deltaLabel(state.resources.money, previous.resources.money, ["money", "金钱"]),
    );
    setText("[data-replay-hands]", state.resources.hands_left);
    setText("[data-replay-discards]", state.resources.discards_left);

    setText(
      "[data-replay-blind-kind]",
      blind?.boss
        ? language === "zh"
          ? "BOSS 盲注"
          : "BOSS BLIND"
        : (state.progress.blind_on_deck ?? "Blind").toUpperCase(),
    );
    setText("[data-replay-blind-name]", blind?.name ?? "No active blind");
    setText("[data-replay-blind-target]", number.format(blind?.target_chips ?? 0));
    setText(
      "[data-replay-blind-dollar-reward]",
      language === "zh"
        ? `金钱奖励 $${blind?.dollar_reward ?? 0}`
        : `Dollar reward $${blind?.dollar_reward ?? 0}`,
    );
    required("[data-replay-blind-token]").classList.toggle("is-boss", Boolean(blind?.boss));

    setText("[data-replay-hand-name]", state.last_hand.handname || "—");
    setText("[data-replay-hand-chips]", number.format(state.last_hand.chips ?? 0));
    setText("[data-replay-hand-mult]", number.format(state.last_hand.mult ?? 0));
    setText("[data-replay-phase]", localized(phaseCopy[state.phase], state.phase));
    setText(
      "[data-replay-action]",
      localized(actionCopy[frame.action?.kind ?? "initial_state"], frame.action?.kind ?? "Initial state"),
    );
    setText("[data-replay-action-detail]", actionDetail(frame.action, frame));

    renderCardRow("[data-replay-jokers]", state.jokers, state.resources.joker_slots);
    renderCardRow("[data-replay-hand]", state.hand, state.resources.hand_size);
    renderCardRow(
      "[data-replay-consumables]",
      state.consumables,
      state.resources.consumable_slots,
      true,
    );
    setText("[data-replay-joker-count]", `${state.jokers.length} / ${state.resources.joker_slots}`);
    setText(
      "[data-replay-hand-count]",
      language === "zh"
        ? `${state.hand.length} 张`
        : `${state.hand.length} card${state.hand.length === 1 ? "" : "s"}`,
    );
    setText(
      "[data-replay-consumable-count]",
      `${state.consumables.length} / ${state.resources.consumable_slots}`,
    );
    renderMarket(state);

    previousButton.disabled = index === 0;
    nextButton.disabled = index === frames.length - 1;
    eventTrack.querySelectorAll("button").forEach((button, frameIndex) => {
      button.classList.toggle("is-current", frameIndex === index);
      button.classList.toggle("is-past", frameIndex < index);
    });
    eventTrack.querySelector<HTMLElement>(".is-current")?.scrollIntoView({
      block: "nearest",
      inline: "center",
      behavior: "smooth",
    });
  }

  function schedule(): void {
    if (!playing) return;
    if (index >= frames.length - 1) {
      pause();
      return;
    }
    const delay = 1050 / Number(speed.value);
    timer = window.setTimeout(() => {
      index += 1;
      render();
      schedule();
    }, delay);
  }

  function play(): void {
    if (!frames.length) return;
    if (index >= frames.length - 1) index = 0;
    playing = true;
    playButton.classList.add("is-playing");
    playSymbol.textContent = "Ⅱ";
    playLabel.textContent = language === "zh" ? "暂停" : "Pause";
    schedule();
  }

  function pause(): void {
    playing = false;
    if (timer !== null) window.clearTimeout(timer);
    timer = null;
    playButton.classList.remove("is-playing");
    playSymbol.textContent = "▶";
    playLabel.textContent = language === "zh" ? "播放" : "Play";
  }

  async function loadText(text: string, label: string): Promise<void> {
    pause();
    try {
      const parsed = parseReplay(text);
      episode = parsed.episode;
      frames = parsed.frames;
      index = 0;
      sourceName.textContent = label;
      status.textContent =
        language === "zh"
          ? `${frames.length - 1} 步 · 分数 ${number.format(episode.score)} · ${episode.status}`
          : `${frames.length - 1} steps · score ${number.format(episode.score)} · ${episode.status}`;
      renderEvents();
      render();
    } catch (error) {
      status.textContent =
        error instanceof Error
          ? error.message
          : language === "zh"
            ? "无法读取回放"
            : "Could not read replay";
    }
  }

  async function loadBundled(): Promise<void> {
    sourceName.textContent = language === "zh" ? "内置 Baseline 回放" : "Bundled baseline replay";
    status.textContent = language === "zh" ? "正在载入…" : "Loading…";
    try {
      const response = await fetch(bundledSource);
      if (!response.ok) throw new Error(`Replay request failed (${response.status})`);
      await loadText(await response.text(), language === "zh" ? "Baseline 回放" : "Baseline replay");
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : "Could not load replay";
    }
  }

  playButton.addEventListener("click", () => (playing ? pause() : play()));
  previousButton.addEventListener("click", () => {
    pause();
    index = Math.max(0, index - 1);
    render();
  });
  nextButton.addEventListener("click", () => {
    pause();
    index = Math.min(frames.length - 1, index + 1);
    render();
  });
  range.addEventListener("input", () => {
    pause();
    index = Number(range.value);
    render();
  });
  speed.addEventListener("change", () => {
    if (playing) {
      if (timer !== null) window.clearTimeout(timer);
      schedule();
    }
  });
  openButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (file) await loadText(await file.text(), file.name);
  });
  resetButton.addEventListener("click", loadBundled);

  for (const eventName of ["dragenter", "dragover"]) {
    dropTarget.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropTarget.classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    dropTarget.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropTarget.classList.remove("is-dragging");
    });
  }
  dropTarget.addEventListener("drop", async (event) => {
    const file = event.dataTransfer?.files[0];
    if (file) await loadText(await file.text(), file.name);
  });

  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    if (event.key === " ") {
      event.preventDefault();
      playing ? pause() : play();
    } else if (event.key === "ArrowLeft") {
      pause();
      index = Math.max(0, index - 1);
      render();
    } else if (event.key === "ArrowRight") {
      pause();
      index = Math.min(frames.length - 1, index + 1);
      render();
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pause();
  });
  window.addEventListener("epg:language", (event) => {
    const nextLanguage = (event as CustomEvent<{ language: "en" | "zh" }>).detail.language;
    language = nextLanguage;
    if (frames.length) {
      renderEvents();
      render();
      if (episode) {
        status.textContent =
          language === "zh"
            ? `${frames.length - 1} 步 · 分数 ${number.format(episode.score)} · ${episode.status}`
            : `${frames.length - 1} steps · score ${number.format(episode.score)} · ${episode.status}`;
      }
    }
    pause();
  });

  void loadBundled();
}
