// Typed client for the Studio notifications lane (/api/studio/notify).
//
// The feed is a bounded ledger (newest first, max 500). `after` polls
// incrementally — `unread` / `latest_id` always describe the whole ring, so
// the bell badge stays correct even when only deltas are requested.

import { api } from "./api";

export interface NotificationItem {
  id: number;
  kind: string;
  title: string;
  body: string;
  link: string;
  created_at: string;
  read: boolean;
}

export interface NotifySettings {
  forward_telegram: boolean;
}

export interface NotifyFeed {
  items: NotificationItem[];
  unread: number;
  latest_id: number;
  settings: NotifySettings;
}

export const fetchNotifications = (after = 0) =>
  api.get<NotifyFeed>(`/notify?after=${encodeURIComponent(after)}`);

export const markNotificationRead = (id: number) =>
  api.post<{ ok: boolean }>(`/notify/${encodeURIComponent(id)}/read`, {});

export const markAllNotificationsRead = () =>
  api.post<{ ok: boolean; marked: number }>("/notify/read-all", {});

export const setNotifySettings = (settings: NotifySettings) =>
  api.post<{ settings: NotifySettings }>("/notify/settings", settings);
