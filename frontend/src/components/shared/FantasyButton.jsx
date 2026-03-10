import React from "react";

const variants = {
  primary:
    "nav-glyph-btn is-active border-[var(--gold)] text-[var(--candle-glow)] hover:border-[var(--gold)] hover:text-[var(--candle-glow)]",
  secondary: "nav-glyph-btn",
  ghost:
    "nav-glyph-btn border-transparent bg-transparent hover:bg-[rgba(255,210,122,0.1)] hover:border-[#5c4228]",
  danger:
    "nav-glyph-btn border-[#7a2020] text-[#c05050] hover:border-[#9a3030] hover:text-[#e87878] hover:shadow-[0_0_8px_rgba(192,80,80,0.35)]",
};

/**
 * Button with fantasy GM-desk variants: primary, secondary, ghost, danger.
 */
export default function FantasyButton({
  variant = "secondary",
  type = "button",
  disabled = false,
  className = "",
  children,
  ...rest
}) {
  const base = "cursor-pointer transition-all duration-200 disabled:opacity-60 disabled:cursor-not-allowed";
  const variantClass = variants[variant] ?? variants.secondary;
  return (
    <button
      type={type}
      disabled={disabled}
      className={`${base} ${variantClass} ${className}`.trim()}
      {...rest}
    >
      {children}
    </button>
  );
}
