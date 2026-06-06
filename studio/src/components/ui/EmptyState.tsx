import type { ReactNode } from "react";

// A centered empty/zero state — generalizes the old .chat-hero pattern for reuse
// across Home, Connections, Approvals, and "coming soon" placeholders.
export function EmptyState({
  icon,
  title,
  children,
  action,
}: {
  icon?: ReactNode;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      {icon && <div className="empty-icon">{icon}</div>}
      <div className="empty-title">{title}</div>
      {children && <div className="empty-sub">{children}</div>}
      {action && <div className="empty-action">{action}</div>}
    </div>
  );
}
