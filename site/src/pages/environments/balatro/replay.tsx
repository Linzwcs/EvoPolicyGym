import {AcademicPage} from "../../../components/AcademicPage";
import {Localized, useSiteLanguage} from "../../../components/Localized";
import {ReplayViewer} from "../../../features/replay/ReplayViewer";

export default function BalatroReplayPage() {
  const language = useSiteLanguage();
  return (
    <AcademicPage
      title={language === "zh" ? "Balatro Policy 回放" : "Balatro Policy replay"}
      description="Inspect a bounded semantic JSONL trace from the EvoPolicyGym Balatro Benchmark."
      eyebrow={<Localized en="Interactive research artifact" zh="交互式研究产物" />}
      heading={<Localized en="Balatro Policy replay" zh="Balatro Policy 回放" />}
      lead={
        <p>
          <Localized
            en="Inspect the state, action, economy, hand, Jokers, and shop decisions in a bounded semantic trace. Open a local JSONL file or use the bundled baseline."
            zh="检查有界语义 trace 中的状态、Action、经济、手牌、Joker 与商店决策；可以打开本地 JSONL，也可以使用内置 baseline。"
          />
        </p>
      }
      meta={
        <dl>
          <div><dt><Localized en="Format" zh="格式" /></dt><dd>JSONL</dd></div>
          <div><dt><Localized en="Input" zh="输入" /></dt><dd><Localized en="Local file or bundled trace" zh="本地文件或内置 trace" /></dd></div>
          <div><dt><Localized en="Execution" zh="执行" /></dt><dd><Localized en="Browser only" zh="仅浏览器" /></dd></div>
        </dl>
      }
      className="replay-page"
    >
      <div className="epg-wide replay-intro-note">
        <span>SPACE</span>
        <Localized
          en="toggles playback; arrow keys step through decisions."
          zh="控制播放；方向键逐步查看决策。"
        />
      </div>
      <div className="epg-wide">
        <ReplayViewer />
      </div>
    </AcademicPage>
  );
}
