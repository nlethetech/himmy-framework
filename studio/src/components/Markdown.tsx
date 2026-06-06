import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Render agent output as formatted content — GFM tables, headings, bold, lists,
// code. react-markdown is XSS-safe by default (no raw HTML injection). The `.md`
// wrapper carries the One-Dark styling (see index.css). Links open in a new tab.
export function Markdown({ children }: { children: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: (props) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
