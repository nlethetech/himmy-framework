import { Link } from "react-router-dom";
import {
  MailIcon,
  TelegramIcon,
  GlobeIcon,
  MemoryIcon,
  BookIcon,
} from "./icons";

// A friendly capability → the tool_packs it switches on + an optional connection
// it needs. Toggling a capability edits form.tool_packs (the real spec field).
export interface Capability {
  key: string;
  label: string;
  description: string;
  hue: string;
  packs: string[];
  needs?: string; // connection type required to actually work
  Icon: (p: { className?: string }) => JSX.Element;
}

export const CAPABILITIES: Capability[] = [
  {
    key: "email",
    label: "Email",
    description: "Send email on your behalf",
    hue: "email",
    packs: ["comms"],
    needs: "email",
    Icon: MailIcon,
  },
  {
    key: "telegram",
    label: "Telegram",
    description: "Message a Telegram chat",
    hue: "telegram",
    packs: ["telegram"],
    needs: "telegram",
    Icon: TelegramIcon,
  },
  {
    key: "web",
    label: "Web",
    description: "Search and read the web",
    hue: "web",
    packs: ["web"],
    needs: "web_search",
    Icon: GlobeIcon,
  },
  {
    key: "memory",
    label: "Memory",
    description: "Remember and recall facts",
    hue: "memory",
    packs: ["memory"],
    Icon: MemoryIcon,
  },
  {
    key: "knowledge",
    label: "Knowledge",
    description: "Search your documents (RAG)",
    hue: "knowledge",
    packs: ["knowledge"],
    Icon: BookIcon,
  },
];

export function CapabilityToggle({
  cap,
  on,
  connected,
  onToggle,
}: {
  cap: Capability;
  on: boolean;
  connected: boolean | null; // null = no connection needed
  onToggle: () => void;
}) {
  return (
    <div className={"cap-row" + (on ? " on" : "")}>
      <span
        className="cap-icon"
        style={{
          background: `color-mix(in srgb, var(--hue-${cap.hue}) 16%, transparent)`,
          color: `var(--hue-${cap.hue})`,
        }}
      >
        <cap.Icon />
      </span>
      <div className="cap-text">
        <div className="cap-label">{cap.label}</div>
        <div className="cap-desc">{cap.description}</div>
        {on && cap.needs && connected === false && (
          <Link className="cap-need" to="/connections">
            ⚠ Needs a {cap.label} connection — connect
          </Link>
        )}
      </div>
      <button
        className={"switch" + (on ? " on" : "")}
        onClick={onToggle}
        role="switch"
        aria-checked={on}
        aria-label={cap.label}
      >
        <span className="switch-knob" />
      </button>
    </div>
  );
}
