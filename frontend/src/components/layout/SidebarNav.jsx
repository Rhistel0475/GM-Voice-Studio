import React from "react";
import { NavLink } from "react-router-dom";
import { Book, Mic, Users, Volume2, LayoutDashboard, Settings } from "lucide-react";

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
    <nav className="flex flex-col gap-1 p-2">
      {navItems.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          className={({ isActive }) =>
            `nav-glyph-btn flex items-center gap-2 w-full justify-start ${isActive ? "is-active" : ""}`
          }
        >
          <Icon size={16} />
          {label}
        </NavLink>
      ))}
    </nav>
  );
}
