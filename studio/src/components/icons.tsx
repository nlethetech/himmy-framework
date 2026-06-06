// Minimal inline SVG icons (stroke-based, inherit currentColor). Keeps the bundle
// dependency-free and the visual language consistent.

type P = { className?: string };
const base = {
  width: 18,
  height: 18,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.7,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export const ChatIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
  </svg>
);

export const BuildIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18v3h3l6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.1-.5-.5-2.1z" />
  </svg>
);

export const RunsIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <line x1="8" y1="6" x2="21" y2="6" />
    <line x1="8" y1="12" x2="21" y2="12" />
    <line x1="8" y1="18" x2="21" y2="18" />
    <circle cx="3.5" cy="6" r="1.3" />
    <circle cx="3.5" cy="12" r="1.3" />
    <circle cx="3.5" cy="18" r="1.3" />
  </svg>
);

export const DoctorIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
  </svg>
);

export const SendIcon = (p: P) => (
  <svg {...base} className={p.className} width={16} height={16}>
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

export const CheckIcon = (p: P) => (
  <svg {...base} className={p.className} width={14} height={14}>
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

export const XIcon = (p: P) => (
  <svg {...base} className={p.className} width={14} height={14}>
    <line x1="18" y1="6" x2="6" y2="18" />
    <line x1="6" y1="6" x2="18" y2="18" />
  </svg>
);

export const PlusIcon = (p: P) => (
  <svg {...base} className={p.className} width={15} height={15}>
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

export const BackIcon = (p: P) => (
  <svg {...base} className={p.className} width={16} height={16}>
    <line x1="19" y1="12" x2="5" y2="12" />
    <polyline points="12 19 5 12 12 5" />
  </svg>
);

export const RefreshIcon = (p: P) => (
  <svg {...base} className={p.className} width={15} height={15}>
    <polyline points="23 4 23 10 17 10" />
    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
  </svg>
);

export const HomeIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <path d="M3 9.5 12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1z" />
  </svg>
);

export const BellIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9" />
    <path d="M13.7 21a2 2 0 0 1-3.4 0" />
  </svg>
);

export const PlugIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <path d="M12 22v-5" />
    <path d="M9 8V2M15 8V2" />
    <path d="M7 8h10v3a5 5 0 0 1-10 0z" />
  </svg>
);

export const ChevronIcon = (p: P) => (
  <svg {...base} className={p.className} width={14} height={14}>
    <polyline points="9 6 15 12 9 18" />
  </svg>
);

export const MailIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <rect x="2" y="4" width="20" height="16" rx="2" />
    <path d="m22 7-10 6L2 7" />
  </svg>
);

export const TelegramIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <path d="M21.5 4.3 2.9 11.5c-1 .4-1 1 .1 1.3l4.6 1.4 1.8 5.4c.2.6.1.9.7.9.5 0 .7-.2 1-.5l2.4-2.3 4.7 3.5c.9.5 1.5.2 1.7-.8l3.1-14.6c.3-1.2-.5-1.8-1.3-1.4z" />
  </svg>
);

export const GlobeIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <circle cx="12" cy="12" r="9" />
    <path d="M3 12h18M12 3a14 14 0 0 1 0 18 14 14 0 0 1 0-18" />
  </svg>
);

export const MemoryIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <path d="M12 3a5 5 0 0 0-5 5c0 1-1 1.5-1 3a3 3 0 0 0 3 3v3a2 2 0 0 0 4 0 2 2 0 0 0 4 0v-3a3 3 0 0 0 3-3c0-1.5-1-2-1-3a5 5 0 0 0-5-5 3 3 0 0 0-3 0z" />
  </svg>
);

export const BookIcon = (p: P) => (
  <svg {...base} className={p.className}>
    <path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v18H6.5A2.5 2.5 0 0 0 4 22.5z" />
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
  </svg>
);
