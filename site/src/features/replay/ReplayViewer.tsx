import useBaseUrl from "@docusaurus/useBaseUrl";
import {
  type ChangeEvent,
  type DragEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {Localized, useSiteLanguage} from "../../components/Localized";
import {
  parseReplay,
  type ParsedReplay,
  type ReplayAction,
  type ReplayCard,
  type ReplayFrame,
} from "./model";

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

const actionCopy: Record<string, {en: string; zh: string}> = {
  initial_state: {en: "Initial state", zh: "初始状态"},
  select_blind: {en: "Enter blind", zh: "进入盲注"},
  skip_blind: {en: "Skip blind", zh: "跳过盲注"},
  play_hand: {en: "Play hand", zh: "出牌"},
  discard: {en: "Discard", zh: "弃牌"},
  cash_out: {en: "Cash out", zh: "领取奖励"},
  reroll_shop: {en: "Reroll shop", zh: "刷新商店"},
  next_round: {en: "Leave shop", zh: "离开商店"},
  buy_card: {en: "Buy card", zh: "购买卡牌"},
  sell_joker: {en: "Sell Joker", zh: "出售 Joker"},
  sell_consumable: {en: "Sell consumable", zh: "出售消耗牌"},
  use_consumable: {en: "Use consumable", zh: "使用消耗牌"},
  redeem_voucher: {en: "Redeem voucher", zh: "兑换优惠券"},
  open_booster: {en: "Open booster", zh: "打开补充包"},
  pick_pack_card: {en: "Choose pack card", zh: "选择补充包卡牌"},
  skip_pack: {en: "Skip pack", zh: "跳过补充包"},
};

const phaseCopy: Record<string, {en: string; zh: string}> = {
  blind_select: {en: "Blind select", zh: "选择盲注"},
  selecting_hand: {en: "Playing blind", zh: "盲注进行中"},
  round_eval: {en: "Round complete", zh: "本轮完成"},
  shop: {en: "Shop", zh: "商店"},
  booster_pack: {en: "Booster pack", zh: "补充包"},
  game_over: {en: "Run over", zh: "本局结束"},
};

export function ReplayViewer() {
  const language = useSiteLanguage();
  const bundledSource = useBaseUrl("data/balatro-baseline-replay.json");
  const [replay, setReplay] = useState<ParsedReplay | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [sourceName, setSourceName] = useState(
    language === "zh" ? "内置 Baseline 回放" : "Bundled baseline replay",
  );
  const [status, setStatus] = useState(language === "zh" ? "正在载入…" : "Loading…");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const loadText = useCallback(
    (text: string, label: string) => {
      setPlaying(false);
      try {
        const parsed = parseReplay(text);
        setReplay(parsed);
        setFrameIndex(0);
        setSourceName(label);
        setStatus(
          language === "zh"
            ? `${parsed.frames.length - 1} 步 · 分数 ${parsed.episode.score} · ${parsed.episode.status}`
            : `${parsed.frames.length - 1} steps · score ${parsed.episode.score} · ${parsed.episode.status}`,
        );
      } catch (error) {
        setStatus(error instanceof Error ? error.message : "Could not read replay");
      }
    },
    [language],
  );

  const loadBundled = useCallback(async () => {
    setPlaying(false);
    setSourceName(
      language === "zh" ? "内置 Baseline 回放" : "Bundled baseline replay",
    );
    setStatus(language === "zh" ? "正在载入…" : "Loading…");
    try {
      const response = await fetch(bundledSource);
      if (!response.ok) {
        throw new Error(`Replay request failed (${response.status})`);
      }
      loadText(
        await response.text(),
        language === "zh" ? "Baseline 回放" : "Baseline replay",
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not load replay");
    }
  }, [bundledSource, language, loadText]);

  useEffect(() => {
    void loadBundled();
  }, [loadBundled]);

  useEffect(() => {
    if (!playing || !replay) return;
    if (frameIndex >= replay.frames.length - 1) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(
      () => setFrameIndex((current) => current + 1),
      1050 / speed,
    );
    return () => window.clearTimeout(timer);
  }, [frameIndex, playing, replay, speed]);

  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLSelectElement ||
        event.target instanceof HTMLButtonElement
      ) {
        return;
      }
      if (event.key === " ") {
        event.preventDefault();
        setPlaying((current) => !current);
      } else if (event.key === "ArrowLeft") {
        setPlaying(false);
        setFrameIndex((current) => Math.max(0, current - 1));
      } else if (event.key === "ArrowRight" && replay) {
        setPlaying(false);
        setFrameIndex((current) =>
          Math.min(replay.frames.length - 1, current + 1),
        );
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [replay]);

  const frame = replay?.frames[frameIndex] ?? null;
  const previous =
    replay?.frames[Math.max(0, frameIndex - 1)] ?? frame ?? null;

  const openFile = async (file: File | undefined) => {
    if (file) loadText(await file.text(), file.name);
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    void openFile(event.dataTransfer.files[0]);
  };

  return (
    <section
      className={`replay-lab ${dragging ? "is-dragging" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <header className="replay-toolbar">
        <div className="replay-controls">
          <button
            type="button"
            aria-label={language === "zh" ? "上一步" : "Previous step"}
            disabled={frameIndex === 0}
            onClick={() => {
              setPlaying(false);
              setFrameIndex((current) => Math.max(0, current - 1));
            }}
          >
            ←
          </button>
          <button
            type="button"
            className="replay-play"
            onClick={() => {
              if (replay && frameIndex >= replay.frames.length - 1) {
                setFrameIndex(0);
              }
              setPlaying((current) => !current);
            }}
          >
            {playing ? "Ⅱ" : "▶"}{" "}
            <span>{playing ? (language === "zh" ? "暂停" : "Pause") : language === "zh" ? "播放" : "Play"}</span>
          </button>
          <button
            type="button"
            aria-label={language === "zh" ? "下一步" : "Next step"}
            disabled={!replay || frameIndex >= replay.frames.length - 1}
            onClick={() => {
              setPlaying(false);
              setFrameIndex((current) =>
                Math.min((replay?.frames.length ?? 1) - 1, current + 1),
              );
            }}
          >
            →
          </button>
        </div>
        <div className="replay-scrubber">
          <span>{String(frameIndex).padStart(2, "0")} / {String((replay?.frames.length ?? 1) - 1).padStart(2, "0")}</span>
          <input
            type="range"
            min="0"
            max={Math.max(0, (replay?.frames.length ?? 1) - 1)}
            value={frameIndex}
            onChange={(event) => {
              setPlaying(false);
              setFrameIndex(Number(event.target.value));
            }}
          />
          <select
            aria-label={language === "zh" ? "播放速度" : "Playback speed"}
            value={speed}
            onChange={(event) => setSpeed(Number(event.target.value))}
          >
            <option value="0.5">0.5×</option>
            <option value="1">1×</option>
            <option value="2">2×</option>
            <option value="4">4×</option>
          </select>
        </div>
        <div className="replay-file-actions">
          <button type="button" onClick={() => fileInput.current?.click()}>
            <Localized en="Open JSONL" zh="打开 JSONL" />
          </button>
          <input
            ref={fileInput}
            hidden
            type="file"
            accept=".jsonl,.ndjson,application/x-ndjson"
            onChange={(event: ChangeEvent<HTMLInputElement>) =>
              void openFile(event.target.files?.[0])
            }
          />
        </div>
      </header>

      {frame && previous ? (
        <ReplayBoard
          frame={frame}
          previous={previous}
          language={language}
        />
      ) : (
        <div className="replay-loading">{status}</div>
      )}

      <footer className="replay-source">
        <div>
          <span>JSONL</span>
          <strong>{sourceName}</strong>
          <small>{status}</small>
        </div>
        <button type="button" onClick={() => void loadBundled()}>
          <Localized en="Reset baseline" zh="重置 Baseline" />
        </button>
      </footer>

      {replay && (
        <div className="replay-events" aria-label="Replay timeline">
          {replay.frames.map((item, index) => {
            const action = item.action?.kind ?? "initial_state";
            const label = actionCopy[action]?.[language] ?? action;
            return (
              <button
                type="button"
                key={`${index}-${action}`}
                className={[
                  index === frameIndex ? "is-current" : "",
                  index < frameIndex ? "is-past" : "",
                ].join(" ")}
                aria-label={`${index}: ${label}`}
                title={label}
                onClick={() => {
                  setPlaying(false);
                  setFrameIndex(index);
                }}
              />
            );
          })}
        </div>
      )}
    </section>
  );
}

function ReplayBoard({
  frame,
  previous,
  language,
}: {
  frame: ReplayFrame;
  previous: ReplayFrame;
  language: "en" | "zh";
}) {
  const state = frame.state;
  const market =
    state.pack.cards.length > 0
      ? state.pack.cards
      : [...state.shop.cards, ...state.shop.vouchers, ...state.shop.boosters];
  const action = frame.action?.kind ?? "initial_state";
  const actionLabel = actionCopy[action]?.[language] ?? action.replaceAll("_", " ");
  const phase = phaseCopy[state.phase]?.[language] ?? state.phase.replaceAll("_", " ");

  return (
    <div className="replay-board">
      <div className="replay-stats">
        <Metric
          label={language === "zh" ? "进度" : "Progress"}
          value={`Ante ${state.progress.ante} / ${state.progress.win_ante}`}
          detail={
            language === "zh"
              ? `已通过 ${state.progress.rounds_cleared} 个盲注`
              : `${state.progress.rounds_cleared} blinds cleared`
          }
        />
        <Metric
          label={language === "zh" ? "分数" : "Score"}
          value={`${state.resources.chips} / ${state.blind?.target_chips ?? 0}`}
          detail={delta(
            state.resources.chips,
            previous.state.resources.chips,
            language === "zh" ? "分数" : "score",
          )}
        />
        <Metric
          label={language === "zh" ? "金钱" : "Money"}
          value={`$${state.resources.money}`}
          detail={delta(
            state.resources.money,
            previous.state.resources.money,
            language === "zh" ? "金钱" : "money",
          )}
        />
        <Metric
          label={language === "zh" ? "资源" : "Resources"}
          value={`${state.resources.hands_left} H · ${state.resources.discards_left} D`}
          detail={phase}
        />
      </div>

      <div className="replay-stage">
        <aside className={`replay-blind ${state.blind?.boss ? "is-boss" : ""}`}>
          <small>
            {state.blind?.boss
              ? language === "zh" ? "BOSS 盲注" : "BOSS BLIND"
              : (state.progress.blind_on_deck ?? "BLIND").toUpperCase()}
          </small>
          <strong>{state.blind?.name ?? "No active blind"}</strong>
          <b>{state.blind?.target_chips ?? 0}</b>
          <span>${state.blind?.dollar_reward ?? 0}</span>
        </aside>

        <section className="replay-center">
          <header>
            <span>{phase}</span>
            <strong>{actionLabel}</strong>
            <small>{actionDetail(frame.action, language)}</small>
          </header>

          <CardSection
            label="Jokers"
            count={`${state.jokers.length} / ${state.resources.joker_slots}`}
            cards={state.jokers}
            slots={state.resources.joker_slots}
          />
          <CardSection
            label={language === "zh" ? "手牌" : "Hand"}
            count={`${state.hand.length}`}
            cards={state.hand}
            slots={state.resources.hand_size}
          />
        </section>

        <aside className="replay-side">
          <div>
            <small>{language === "zh" ? "上一手" : "Last hand"}</small>
            <strong>{state.last_hand.handname || "—"}</strong>
            <code>{state.last_hand.chips ?? 0} × {state.last_hand.mult ?? 0}</code>
          </div>
          <div>
            <small>{state.pack.cards.length ? (language === "zh" ? "补充包" : "Pack") : language === "zh" ? "商店" : "Shop"}</small>
            {market.length ? (
              market.slice(0, 6).map((card) => (
                <span key={`${card.index}-${card.name}`}>
                  <b>{card.name}</b>
                  <i>{card.cost ? `$${card.cost}` : card.set}</i>
                </span>
              ))
            ) : (
              <p>{language === "zh" ? "当前没有可见商品" : "No visible items"}</p>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function CardSection({
  label,
  count,
  cards,
  slots,
}: {
  label: string;
  count: string;
  cards: ReplayCard[];
  slots: number;
}) {
  const emptySlots = Math.max(0, Math.min(slots - cards.length, 5));
  return (
    <section className="replay-card-section">
      <header><strong>{label}</strong><small>{count}</small></header>
      <div className="replay-card-row">
        {cards.map((card) => <ReplayCardView key={`${card.index}-${card.name}`} card={card} />)}
        {Array.from({length: emptySlots}, (_, index) => (
          <span className="replay-card-empty" key={`empty-${index}`}>+</span>
        ))}
      </div>
    </section>
  );
}

function ReplayCardView({card}: {card: ReplayCard}) {
  const playingCard = Boolean(card.rank && card.suit);
  const faceDown = card.facing === "back";
  const suit = suitSymbols[card.suit ?? ""] ?? "?";
  const rank = rankSymbols[card.rank ?? ""] ?? card.rank ?? "?";
  const red = card.suit === "Hearts" || card.suit === "Diamonds";
  const ability = useMemo(
    () =>
      Object.entries(card.ability ?? {}).find(
        ([, value]) =>
          (typeof value === "number" && value !== 0 && value !== 1) ||
          typeof value === "string",
      ),
    [card.ability],
  );

  if (faceDown) {
    return <article className="replay-card replay-card--back">EPG</article>;
  }
  if (playingCard) {
    return (
      <article className={`replay-card replay-card--playing ${red ? "is-red" : ""}`}>
        <span><b>{rank}</b><i>{suit}</i></span>
        <strong>{suit}</strong>
        <small>{card.chips ?? 0}</small>
      </article>
    );
  }
  return (
    <article className="replay-card replay-card--entity" title={card.name}>
      <span>{card.set || "Card"}</span>
      <strong>{card.name}</strong>
      <small>{ability ? `${ability[0]} ${String(ability[1])}` : card.cost ? `$${card.cost}` : "—"}</small>
    </article>
  );
}

function delta(current: number, previous: number, noun: string): string {
  const value = current - previous;
  if (value === 0) return `No ${noun} change`;
  return `${value > 0 ? "+" : ""}${value} ${noun}`;
}

function actionDetail(action: ReplayAction | null, language: "en" | "zh") {
  if (!action) return language === "zh" ? "等待第一次决策" : "Waiting for first decision";
  if (action.card_indices?.length) {
    const cards = action.card_indices.map((value) => `#${value + 1}`).join(", ");
    return language === "zh" ? `卡牌 ${cards}` : `Cards ${cards}`;
  }
  if (typeof action.target_index === "number") {
    return language === "zh"
      ? `目标 #${action.target_index + 1}`
      : `Target #${action.target_index + 1}`;
  }
  return language === "zh" ? "状态已更新" : "State updated";
}
