import { useCallback, useEffect, useRef, useState } from "react";
import { Topbar } from "../components/Page";
import { Modal } from "../components/ui/Modal";
import { PickMenu } from "../components/ui/PickMenu";
import { useToast } from "../components/ui/Toast";
import { relativeTime } from "../lib/format";
import {
  erasePrivacySubject,
  exportAuditBundle,
  getPrivacyConsents,
  getPrivacySubjects,
  getSeclog,
  verifyAuditBundle,
  type ConsentEntry,
  type PrivacySubject,
  type SeclogEvent,
  type VerifyResult,
} from "../lib/privacyApi";
import "../styles/privacy.css";

/* Privacy as a single centered ledger column: SUBJECTS (who the system knows,
   with an erase action), CONSENTS (the append-only version chains), AUDIT
   (export/verify the signed bundle). No cards — micro-labels over hairlines,
   rows beneath. Red is reserved for the destroy action, failures, and
   erased/denied states. */

const when = (iso: string | null | undefined) => {
  if (!iso) return "";
  try {
    return relativeTime(iso);
  } catch {
    return iso.slice(0, 10);
  }
};

export default function Privacy() {
  const toast = useToast();

  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [governed, setGoverned] = useState(true);
  const [subjects, setSubjects] = useState<PrivacySubject[]>([]);
  const [consents, setConsents] = useState<ConsentEntry[]>([]);
  const [filter, setFilter] = useState(""); // "" = all subjects

  const [eraseTarget, setEraseTarget] = useState<PrivacySubject | null>(null);
  const [confirmText, setConfirmText] = useState("");
  const [erasing, setErasing] = useState(false);

  const [exporting, setExporting] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [verdict, setVerdict] = useState<VerifyResult | null>(null);
  const [auditNote, setAuditNote] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // ---- security events (seclog) — mirrors `himmy seclog` -----------------
  const [seclog, setSeclog] = useState<SeclogEvent[]>([]);
  const [seclogTypes, setSeclogTypes] = useState<string[]>([]);
  const [seclogType, setSeclogType] = useState(""); // "" = all event types
  const [seclogError, setSeclogError] = useState<string | null>(null);

  const loadSeclog = useCallback(async (type?: string) => {
    setSeclogError(null);
    try {
      const res = await getSeclog({ limit: 100, type: type || undefined });
      setSeclog(res.items);
      // Keep the filter options stable from the unfiltered window the server returns.
      setSeclogTypes((prev) =>
        res.types.length ? res.types : prev,
      );
    } catch (e) {
      setSeclogError(
        e instanceof Error ? e.message : "could not load security events",
      );
    }
  }, []);

  const load = useCallback(async (subject?: string) => {
    setLoadError(null);
    try {
      const [subs, cons] = await Promise.all([
        getPrivacySubjects(),
        getPrivacyConsents(subject || undefined),
      ]);
      setGoverned(subs.governed);
      setSubjects(subs.subjects);
      setConsents(cons.items);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "could not load the ledger");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(filter);
  }, [load, filter]);

  useEffect(() => {
    void loadSeclog(seclogType);
  }, [loadSeclog, seclogType]);

  const startErase = (s: PrivacySubject) => {
    setEraseTarget(s);
    setConfirmText("");
  };

  const doErase = async () => {
    if (!eraseTarget || confirmText !== eraseTarget.subject_id) return;
    setErasing(true);
    try {
      const result = await erasePrivacySubject(
        eraseTarget.subject_id,
        confirmText,
      );
      toast.show(
        result.crypto_shredded
          ? `Erased ${result.subject_id} — key destroyed, data unrecoverable`
          : `Erased ${result.subject_id} — consents withdrawn, tombstone written`,
        "ok",
      );
      setEraseTarget(null);
      void load(filter);
    } catch (e) {
      toast.show(e instanceof Error ? e.message : "erase failed", "err");
    } finally {
      setErasing(false);
    }
  };

  const doExport = async () => {
    setExporting(true);
    setAuditNote(null);
    try {
      const envelope = await exportAuditBundle();
      const blob = new Blob([JSON.stringify(envelope, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `himmy-audit-${envelope.exported_at.replace(/[:.]/g, "-")}.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.show(
        `Bundle exported — ${envelope.record_count} records, ${envelope.algorithm}`,
        "ok",
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : "export failed";
      setAuditNote(msg);
      toast.show(msg, "err");
    } finally {
      setExporting(false);
    }
  };

  const onVerifyFile = async (file: File) => {
    setVerifying(true);
    setVerdict(null);
    setAuditNote(null);
    try {
      let payload: unknown;
      try {
        payload = JSON.parse(await file.text());
      } catch {
        throw new Error("that file is not valid JSON");
      }
      setVerdict(await verifyAuditBundle(payload));
    } catch (e) {
      const msg = e instanceof Error ? e.message : "verification failed";
      setAuditNote(msg);
      toast.show(msg, "err");
    } finally {
      setVerifying(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const filterGroups = [
    {
      options: [
        { value: "", label: "All subjects" },
        ...subjects.map((s) => ({ value: s.subject_id, label: s.subject_id })),
      ],
    },
  ];

  return (
    <>
      <Topbar title="Privacy" sub="consent, retention, and audit" />
      <div className="home-col">
        {/* ---- subjects -------------------------------------------------- */}
        <section className="home-sec">
          <div className="home-sec-head">
            <span>Subjects</span>
            <span className="mono">{subjects.length || ""}</span>
          </div>
          {loading ? (
            <div className="priv-note">loading…</div>
          ) : loadError ? (
            <div className="priv-note priv-fail">
              {loadError} —{" "}
              <button onClick={() => void load(filter)}>retry</button>
            </div>
          ) : subjects.length === 0 ? (
            <div className="priv-note">
              {governed
                ? "No data subjects recorded yet — they appear once a governed run or consent record names one."
                : "Consent governance is off. Set HIMMY_CONSENT=on to record consent, gate retention, and enable erasure."}
            </div>
          ) : (
            subjects.map((s) => (
              <div className="run-line" key={s.subject_id}>
                <span className="run-line-prompt priv-id">{s.subject_id}</span>
                {s.erased && <span className="pill err">erased</span>}
                <span className="run-line-meta mono">
                  {s.consent_records} consents · {s.data_records} records
                  {s.last_activity ? ` · ${when(s.last_activity)}` : ""}
                </span>
                {!s.erased && (
                  <button
                    className="priv-erase"
                    onClick={() => startErase(s)}
                    title="Withdraw all consent and crypto-shred this subject"
                  >
                    erase
                  </button>
                )}
              </div>
            ))
          )}
        </section>

        {/* ---- consents -------------------------------------------------- */}
        <section className="home-sec">
          <div className="home-sec-head">
            <span>Consents</span>
            <span className="mono">{consents.length || ""}</span>
          </div>
          {subjects.length > 0 && (
            <div className="priv-filter-row">
              <PickMenu
                value={filter}
                groups={filterGroups}
                onChange={setFilter}
                placeholder="All subjects"
                title="Filter the ledger by subject"
              />
            </div>
          )}
          {loading ? (
            <div className="priv-note">loading…</div>
          ) : consents.length === 0 ? (
            <div className="priv-note">
              {governed
                ? filter
                  ? `No consent records for ${filter}.`
                  : "No consent records yet — the ledger fills as consent is granted, denied, or withdrawn."
                : "The consent ledger is empty while governance is off."}
            </div>
          ) : (
            consents.map((c) => (
              <div
                className="run-line"
                key={`${c.subject_id}|${c.purpose}|${c.version}`}
              >
                <span className="run-line-prompt">
                  {c.purpose}
                  <span className="run-line-meta mono"> · {c.subject_id}</span>
                </span>
                <span
                  className={
                    "priv-verdict" +
                    (c.effect === "deny" || c.state === "withdrawn"
                      ? " fail"
                      : "")
                  }
                >
                  {c.state} → {c.effect}
                </span>
                <span className="run-line-meta mono">
                  v{c.version} · {when(c.recorded_at) || "—"}
                </span>
              </div>
            ))
          )}
        </section>

        {/* ---- audit ----------------------------------------------------- */}
        <section className="home-sec">
          <div className="home-sec-head">
            <span>Audit</span>
            {verdict && (
              <span className={"mono" + (verdict.ok ? "" : " priv-fail")}>
                {verdict.ok ? "verified" : "verification failed"}
              </span>
            )}
          </div>
          <button className="priv-action" onClick={doExport} disabled={exporting}>
            <span className="priv-action-label">
              {exporting ? "Exporting…" : "Export signed bundle"}
            </span>
            <span className="run-line-meta mono">
              security events · consent · erasure → JSON
            </span>
          </button>
          <button
            className="priv-action"
            onClick={() => fileRef.current?.click()}
            disabled={verifying}
          >
            <span className="priv-action-label">
              {verifying ? "Verifying…" : "Verify bundle"}
            </span>
            <span className="run-line-meta mono">
              upload a bundle → per-check verdicts
            </span>
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".json,application/json"
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void onVerifyFile(f);
            }}
          />
          {auditNote && <div className="priv-note priv-fail">{auditNote}</div>}
          {verdict &&
            verdict.checks.map((chk) => (
              <div className="run-line" key={chk.name}>
                <span className="run-line-prompt">
                  {chk.name}
                  {chk.detail && (
                    <span className="priv-detail">{chk.detail}</span>
                  )}
                </span>
                <span className={"priv-verdict" + (chk.ok ? "" : " fail")}>
                  {chk.ok ? "pass" : "fail"}
                </span>
              </div>
            ))}
        </section>

        {/* ---- security events (seclog) ---------------------------------- */}
        <section className="home-sec">
          <div className="home-sec-head">
            <span>Security events</span>
            <span className="mono">{seclog.length || ""}</span>
          </div>
          {seclogTypes.length > 0 && (
            <div className="priv-filter-row">
              <PickMenu
                value={seclogType}
                groups={[
                  {
                    options: [
                      { value: "", label: "All event types" },
                      ...seclogTypes.map((t) => ({ value: t, label: t })),
                    ],
                  },
                ]}
                onChange={setSeclogType}
                placeholder="All event types"
                title="Filter the security log by event type"
              />
            </div>
          )}
          {seclogError ? (
            <div className="priv-note priv-fail">
              {seclogError} —{" "}
              <button onClick={() => void loadSeclog(seclogType)}>retry</button>
            </div>
          ) : seclog.length === 0 ? (
            <div className="priv-note">
              {seclogType
                ? `No ${seclogType} events recorded.`
                : "No security events recorded — a clean trail. Denied tools, approvals, and authz decisions land here (same as `himmy seclog`)."}
            </div>
          ) : (
            seclog.map((ev) => (
              <div className="run-line" key={ev.event_id}>
                <span className="run-line-prompt">
                  {ev.action || ev.event_type}
                  {ev.resource && (
                    <span className="run-line-meta mono"> · {ev.resource}</span>
                  )}
                </span>
                <span
                  className={
                    "priv-verdict" + (ev.outcome === "deny" ? " fail" : "")
                  }
                >
                  {ev.outcome}
                </span>
                <span className="run-line-meta mono">
                  {ev.actor} · {ev.event_type} · {when(ev.created_at) || "—"}
                </span>
              </div>
            ))
          )}
        </section>
      </div>

      {/* ---- erase confirmation ------------------------------------------ */}
      {eraseTarget && (
        <Modal
          title={`Erase ${eraseTarget.subject_id}`}
          onClose={() => !erasing && setEraseTarget(null)}
          footer={
            <>
              <button
                className="btn"
                onClick={() => setEraseTarget(null)}
                disabled={erasing}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary"
                disabled={confirmText !== eraseTarget.subject_id || erasing}
                onClick={() => void doErase()}
              >
                {erasing ? "Erasing…" : "Erase forever"}
              </button>
            </>
          }
        >
          <p>
            This withdraws every consent and destroys the subject&apos;s
            encryption key. Encrypted records become permanently unrecoverable;
            a signed erasure tombstone remains as proof. This cannot be undone.
          </p>
          <div className="priv-modal-line">
            <span className="k">Consent records to withdraw</span>
            <span className="v">{eraseTarget.consent_records}</span>
          </div>
          <div className="priv-modal-line">
            <span className="k">Data records rendered unreadable</span>
            <span className="v">{eraseTarget.data_records}</span>
          </div>
          <div className="priv-confirm-hint">
            Type {eraseTarget.subject_id} to confirm
          </div>
          <input
            className="input priv-confirm-input"
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            placeholder={eraseTarget.subject_id}
            maxLength={200}
            autoFocus
            onKeyDown={(e) => {
              if (e.key === "Enter") void doErase();
            }}
          />
        </Modal>
      )}
    </>
  );
}
