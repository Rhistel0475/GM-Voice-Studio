export default function SectionHeader({ title, subtitle, icon: Icon, action, className = "" }) {
  return (
    <div
      className={`subhead flex items-center justify-between gap-3 rounded-t ${className}`.trim()}
    >
      <div className="flex items-center gap-2">
        {Icon && <Icon size={16} className="text-[var(--gold)] shrink-0" aria-hidden />}
        <span className="font-heading text-sm">{title}</span>
        {subtitle && <span className="opacity-80 text-xs">({subtitle})</span>}
      </div>
      {action}
    </div>
  );
}
