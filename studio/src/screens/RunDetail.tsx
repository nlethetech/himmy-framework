import { Topbar, Page } from "../components/Page";

export default function RunDetail() {
  return (
    <>
      <Topbar title="Run" sub="trace timeline" />
      <Page>
        <div className="empty">Trace timelines arrive in Phase 2.</div>
      </Page>
    </>
  );
}
