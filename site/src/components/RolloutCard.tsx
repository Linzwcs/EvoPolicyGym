import useBaseUrl from "@docusaurus/useBaseUrl";
import type {Clip} from "../lib/showcase";
import {formatScore} from "../lib/showcase";
import {Localized} from "./Localized";

export function RolloutCard({
  clip,
  rank,
  winner = false,
  compact = false,
}: {
  clip: Clip;
  rank?: number;
  winner?: boolean;
  compact?: boolean;
}) {
  const media = useBaseUrl(clip.media);
  return (
    <article
      className={[
        "research-rollout",
        winner ? "research-rollout--winner" : "",
        compact ? "research-rollout--compact" : "",
      ].join(" ")}
    >
      <div className="research-rollout-media">
        <img
          src={media}
          alt={`${clip.model_display} Policy rerun in ${clip.env_display}`}
          loading="lazy"
        />
        <span>{clip.capture_source.replaceAll("_", " ")}</span>
      </div>
      <div className="research-rollout-body">
        <header>
          <div>
            <h3>{clip.model_display}</h3>
            <small>{clip.harness}</small>
          </div>
          {rank && <b>{winner ? "BEST" : `#${rank}`}</b>}
        </header>
        <dl>
          <div>
            <dt><Localized en="Held-out" zh="Held-out 分数" /></dt>
            <dd>{formatScore(clip.score)}</dd>
          </div>
          <div>
            <dt><Localized en="Checkpoint" zh="Checkpoint" /></dt>
            <dd>#{String(clip.submit_index).padStart(3, "0")}</dd>
          </div>
          <div>
            <dt><Localized en="Rerun" zh="重跑" /></dt>
            <dd>{clip.rerun_steps} steps</dd>
          </div>
        </dl>
        {!compact && clip.event_notes && <p>{clip.event_notes}</p>}
      </div>
    </article>
  );
}
