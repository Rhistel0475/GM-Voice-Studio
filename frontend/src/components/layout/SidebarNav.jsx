import { NavLink } from "react-router-dom";
import { Book, Users, Volume2, LayoutDashboard, Settings } from "lucide-react";

const navItems = [
  { to: "/", label: "Live Board", icon: LayoutDashboard },
  { to: "/codex", label: "Campaign Codex", icon: Book },
  { to: "/npcs", label: "NPC Workshop", icon: Users },
  { to: "/voices", label: "Voice Studio", icon: Volume2 },
  { to: "/prep", label: "Prep Room", icon: Book },
  { to: "/intake", label: "Library", icon: Book },
  { to: "/settings", label: "Settings", icon: Settings },
];

export default function SidebarNav() {
  return (
    <nav className="sidebar-nav">
      {navItems.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          title={label}
          className={({ isActive }) =>
            `nav-glyph-btn flex items-center gap-2 w-full justify-start md:justify-center xl:justify-start md:px-2 ${isActive ? "is-active" : ""}`
          }
        >
          <Icon size={18} className="shrink-0 text-[var(--gold)]/90" aria-hidden />
          <span className="md:sr-only xl:not-sr-only">{label}</span>
        </NavLink>
      ))}
    </nav>
  );
}
