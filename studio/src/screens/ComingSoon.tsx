import { Topbar } from "../components/Page";
import { EmptyState } from "../components/ui/EmptyState";

// Placeholder for screens whose phase hasn't landed yet — keeps the new IA's
// nav fully navigable while the real screens are built incrementally.
export default function ComingSoon({
  title,
  note,
}: {
  title: string;
  note?: string;
}) {
  return (
    <>
      <Topbar title={title} sub="coming soon" />
      <EmptyState icon="✦" title={`${title} is on the way`}>
        {note ?? "This area is being built. It will appear here shortly."}
      </EmptyState>
    </>
  );
}
