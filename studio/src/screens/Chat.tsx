import { Topbar, Page } from "../components/Page";

export default function Chat() {
  return (
    <>
      <Topbar title="Chat" sub="talk to an agent" />
      <Page>
        <div className="empty">Chat arrives in Phase 1.</div>
      </Page>
    </>
  );
}
