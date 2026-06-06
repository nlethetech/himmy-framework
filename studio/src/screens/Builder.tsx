import { Topbar, Page } from "../components/Page";

export default function Builder() {
  return (
    <>
      <Topbar title="Agents" sub="build & edit" />
      <Page>
        <div className="empty">The agent builder arrives in Phase 3.</div>
      </Page>
    </>
  );
}
