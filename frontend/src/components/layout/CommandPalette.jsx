import { useEffect, useRef, useState } from "react";
import { Search } from "lucide-react";
import { ModalShell } from "../shared";

function getShortcutLabel() {
  if (typeof navigator === "undefined") return "Ctrl K";
  return /Mac|iPhone|iPad/.test(navigator.platform) ? "Cmd K" : "Ctrl K";
}

function getNextEnabledIndex(commands, startIndex, step) {
  if (!commands.length) return 0;

  let index = startIndex;
  for (let count = 0; count < commands.length; count += 1) {
    index = (index + step + commands.length) % commands.length;
    if (!commands[index]?.disabled) return index;
  }

  return Math.max(0, startIndex);
}

export default function CommandPalette({
  open,
  onClose,
  commands = [],
}) {
  const inputRef = useRef(null);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const shortcutLabel = getShortcutLabel();
  const normalizedQuery = query.trim().toLowerCase();
  const filteredCommands = normalizedQuery
    ? commands.filter((command) => {
        const haystack = [
          command.title,
          command.description,
          ...(command.keywords || []),
        ]
          .join(" ")
          .toLowerCase();
        return haystack.includes(normalizedQuery);
      })
    : commands;

  useEffect(() => {
    if (!open) {
      setQuery("");
      setActiveIndex(0);
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    });

    return () => window.cancelAnimationFrame(frame);
  }, [open]);

  useEffect(() => {
    if (!open) return;

    const firstEnabledIndex = filteredCommands.findIndex((command) => !command.disabled);
    setActiveIndex((current) => {
      if (!filteredCommands.length) return 0;
      if (current < filteredCommands.length && !filteredCommands[current]?.disabled) return current;
      return firstEnabledIndex >= 0 ? firstEnabledIndex : 0;
    });
  }, [commands, filteredCommands, open]);

  const handleCommandSelect = (command) => {
    if (!command || command.disabled) return;
    command.onSelect?.();
    onClose?.();
  };

  const handleKeyDown = (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose?.();
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((current) => getNextEnabledIndex(filteredCommands, current, 1));
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((current) => getNextEnabledIndex(filteredCommands, current, -1));
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();
      handleCommandSelect(filteredCommands[activeIndex]);
    }
  };

  if (!open) return null;

  return (
    <ModalShell title="Command Palette" onClose={onClose} className="max-w-2xl border border-[#7f5a2c] bg-[#120a05] shadow-[0_30px_100px_rgba(0,0,0,0.55)]">
      <div className="flex flex-col gap-4" onKeyDown={handleKeyDown}>
        <div className="rounded-xl border border-[#5f4122] bg-[linear-gradient(180deg,rgba(34,22,12,0.9),rgba(16,10,6,0.96))] px-4 py-3 shadow-inner">
          <div className="flex items-center gap-3">
            <Search className="h-4 w-4 shrink-0 text-[#d2ab68]" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search commands, tools, scenes..."
              className="w-full border-0 bg-transparent text-sm text-[var(--text-1)] outline-none placeholder:text-[#8c6b42]"
              aria-label="Search commands"
            />
            <span className="hidden rounded-md border border-[#5f4122] bg-[#1a1008] px-2 py-1 text-[10px] font-heading uppercase tracking-[0.22em] text-[#a9874d] sm:inline-flex">
              {shortcutLabel}
            </span>
          </div>
        </div>

        <div className="max-h-[420px] overflow-y-auto pr-1">
          {filteredCommands.length > 0 ? (
            <div className="space-y-2">
              {filteredCommands.map((command, index) => {
                const Icon = command.icon;
                const isActive = index === activeIndex;

                return (
                  <button
                    key={command.id}
                    type="button"
                    className={[
                      "group flex w-full items-start gap-3 rounded-xl border px-3 py-3 text-left transition-all",
                      isActive
                        ? "border-[#c69a52] bg-[linear-gradient(135deg,rgba(73,48,24,0.92),rgba(28,17,10,0.98))] shadow-[0_16px_35px_rgba(0,0,0,0.28)]"
                        : "border-[#4b321b] bg-[rgba(18,10,5,0.78)] hover:border-[#8f6732] hover:bg-[rgba(36,22,12,0.92)]",
                      command.disabled ? "cursor-not-allowed opacity-55" : "cursor-pointer",
                    ].join(" ")}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => handleCommandSelect(command)}
                    disabled={command.disabled}
                  >
                    <div className={`mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border ${isActive ? "border-[#d9b36d] bg-[rgba(222,178,94,0.12)]" : "border-[#5f4122] bg-[rgba(255,255,255,0.02)]"}`}>
                      {Icon ? <Icon className="h-4 w-4 text-[#e3c37d]" /> : null}
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-heading text-sm text-[var(--text-1)]">{command.title}</span>
                        <span className="text-[10px] uppercase tracking-[0.24em] text-[#8f6a39]">
                          {command.group || "Command"}
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-[var(--text-2)]">{command.description}</p>
                      {command.disabledReason ? (
                        <p className="mt-2 text-xs text-[#c69055]">{command.disabledReason}</p>
                      ) : null}
                    </div>

                    <div className="hidden shrink-0 items-center self-center rounded-md border border-[#5f4122] bg-[#130c07] px-2 py-1 text-[10px] font-heading uppercase tracking-[0.22em] text-[#9c7a3a] sm:flex">
                      Enter
                    </div>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-[#5f4122] bg-[rgba(18,10,5,0.75)] px-4 py-8 text-center">
              <div className="font-heading text-sm text-[var(--text-1)]">No commands found</div>
              <p className="mt-2 text-sm text-[var(--text-2)]">
                Try a different search term or clear the filter.
              </p>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-[#3d2815] pt-2 text-[10px] font-heading uppercase tracking-[0.22em] text-[#8f6a39]">
          <span>Arrow Keys Move</span>
          <span>Enter Launches</span>
          <span>Esc Closes</span>
        </div>
      </div>
    </ModalShell>
  );
}
