import type { ReactNode } from "react";

export function Topbar({
  title,
  sub,
  actions,
}: {
  title: string;
  sub?: ReactNode;
  actions?: ReactNode;
}) {
  // The sidebar's active state already says where you are — the title is a
  // whisper for wayfinding, not a header. The sub survives as a tooltip.
  return (
    <div className="topbar">
      <h1 title={typeof sub === "string" ? sub : undefined}>{title}</h1>
      {actions && <div className="row gap10">{actions}</div>}
    </div>
  );
}

export function Page({ children }: { children: ReactNode }) {
  return <div className="page">{children}</div>;
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="center">
      <div className="spinner" />
      <div>{label}</div>
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="banner" style={{ borderColor: "var(--err)" }}>
      <div className="b-ico" style={{ background: "var(--err-soft)", color: "var(--err)" }}>
        !
      </div>
      <div>
        <div className="b-title">Something went wrong</div>
        <div className="b-msg mono">{message}</div>
      </div>
    </div>
  );
}
