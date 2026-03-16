import { FantasyButton } from "../shared";

const TONE_STYLES = {
  gold: {
    frame: "border-[#8b602f]",
    halo: "from-[#e0b66f]/24 via-[#e0b66f]/10 to-transparent",
    icon: "border-[#a97b3f] bg-[rgba(224,182,111,0.14)] text-[#f4d69a]",
    meta: "border-[#7a5a30] text-[#d3ad71]",
  },
  sage: {
    frame: "border-[#597450]",
    halo: "from-[#7ca36f]/24 via-[#7ca36f]/10 to-transparent",
    icon: "border-[#6f8d60] bg-[rgba(124,163,111,0.14)] text-[#d8e7c7]",
    meta: "border-[#4f6845] text-[#b7d0ae]",
  },
  ember: {
    frame: "border-[#8a5036]",
    halo: "from-[#cc7f55]/24 via-[#cc7f55]/10 to-transparent",
    icon: "border-[#a46344] bg-[rgba(204,127,85,0.14)] text-[#f1c4a7]",
    meta: "border-[#78432d] text-[#dbac8d]",
  },
  arcane: {
    frame: "border-[#50627e]",
    halo: "from-[#7b97c7]/24 via-[#7b97c7]/10 to-transparent",
    icon: "border-[#6f86ae] bg-[rgba(123,151,199,0.14)] text-[#d6e1f6]",
    meta: "border-[#445671] text-[#b4c4e4]",
  },
};

export default function VoiceActionCard({
  icon: Icon,
  title,
  description,
  launchLabel = "Launch",
  meta,
  tone = "gold",
  onLaunch,
  disabled = false,
}) {
  const styles = TONE_STYLES[tone] ?? TONE_STYLES.gold;

  return (
    <article className={`panel-ornate relative overflow-hidden rounded-xl min-h-[250px] ${styles.frame}`}>
      <div
        className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${styles.halo}`}
        aria-hidden
      />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-16 bg-[radial-gradient(circle_at_top,rgba(255,245,220,0.12),transparent_70%)]" aria-hidden />
      <div className="relative panel-body h-full gap-4">
        <div className={`flex h-14 w-14 items-center justify-center rounded-full border shadow-[inset_0_0_14px_rgba(0,0,0,0.28)] ${styles.icon}`}>
          <Icon size={24} strokeWidth={1.8} aria-hidden />
        </div>

        <div className="space-y-2">
          <h2 className="font-heading text-xl text-[var(--gold)]">{title}</h2>
          <p className="text-sm leading-relaxed text-[var(--text-2)]">{description}</p>
        </div>

        <div className={`min-h-[72px] whitespace-pre-line rounded-lg border bg-[#120a05]/75 px-3 py-2 text-xs leading-relaxed ${styles.meta}`}>
          {meta}
        </div>

        <FantasyButton
          variant="primary"
          className="mt-auto w-full justify-center"
          onClick={onLaunch}
          disabled={disabled}
        >
          {launchLabel}
        </FantasyButton>
      </div>
    </article>
  );
}
