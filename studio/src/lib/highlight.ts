/* Text-node search highlighting.

   Wraps every case-insensitive occurrence of a query inside a rendered DOM
   subtree in <mark class="hl">. Works on plain text AND rendered markdown
   (it walks text nodes, so element boundaries inside a paragraph are fine —
   only matches that span across two elements are missed, which is the right
   trade-off for a transcript). Used by Chat to honor ?hl= deep links from
   Search. */

export function highlightAll(root: HTMLElement, query: string): HTMLElement[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent || parent.closest("mark.hl, textarea, input, script, style")) {
        return NodeFilter.FILTER_REJECT;
      }
      return node.nodeValue && node.nodeValue.toLowerCase().includes(q)
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_SKIP;
    },
  });

  // Collect first — mutating while walking invalidates the walker.
  const textNodes: Text[] = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode as Text);

  const marks: HTMLElement[] = [];
  for (const node of textNodes) {
    let current: Text = node;
    for (;;) {
      const value = current.nodeValue ?? "";
      const idx = value.toLowerCase().indexOf(q);
      if (idx === -1) break;
      const match = current.splitText(idx);
      const rest = match.splitText(q.length);
      const mark = document.createElement("mark");
      mark.className = "hl";
      match.parentNode?.replaceChild(mark, match);
      mark.appendChild(match);
      marks.push(mark);
      current = rest;
    }
  }
  return marks;
}
