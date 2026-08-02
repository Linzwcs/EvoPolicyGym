export type Scalar = string | number | boolean | null;
export type ReplayValue =
  | Scalar
  | ReplayValue[]
  | {[key: string]: ReplayValue};

export interface ReplayCard {
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

export interface ReplayState {
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
  } | null;
  last_hand: {
    chips?: number;
    mult?: number;
    handname?: string;
  };
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

export interface ReplayAction {
  kind: string;
  card_indices?: number[];
  target_index?: number;
}

export interface ReplayEpisode {
  type: "episode";
  episode_index: number;
  status: string;
  steps: number;
  score: number;
  failure: string | null;
  initial_state: ReplayState;
}

export interface ReplayTransition {
  type: "transition";
  episode_index: number;
  step_index: number;
  action: ReplayAction;
  reward: number;
  state: ReplayState;
  terminated: boolean;
  truncated: boolean;
}

export interface ReplayFrame {
  state: ReplayState;
  action: ReplayAction | null;
  transition: ReplayTransition | null;
}

export interface ParsedReplay {
  episode: ReplayEpisode;
  frames: ReplayFrame[];
}

export function parseReplay(text: string): ParsedReplay {
  const documents = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, lineIndex) => {
      try {
        return JSON.parse(line) as
          | ReplayEpisode
          | ReplayTransition
          | {type: string};
      } catch {
        throw new Error(`Invalid JSON on line ${lineIndex + 1}`);
      }
    });

  const episode = documents.find(
    (document): document is ReplayEpisode => document.type === "episode",
  );
  if (!episode?.initial_state) {
    throw new Error("The replay has no episode header");
  }

  const transitions = documents.filter(
    (document): document is ReplayTransition =>
      document.type === "transition" &&
      "episode_index" in document &&
      document.episode_index === episode.episode_index,
  );
  if (!transitions.length) {
    throw new Error("The replay has no transitions");
  }

  return {
    episode,
    frames: [
      {state: episode.initial_state, action: null, transition: null},
      ...transitions.map((transition) => ({
        state: transition.state,
        action: transition.action,
        transition,
      })),
    ],
  };
}
