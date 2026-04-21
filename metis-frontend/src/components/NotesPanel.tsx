import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeMathjax from "rehype-mathjax";
import { IoClose } from "react-icons/io5";
import { BiSolidFilePdf } from "react-icons/bi";
import type { BBoxSelection } from "./PdfViewer";
import type { EvidenceItem } from "../backend/http";

type EvidenceWithTool = EvidenceItem & { toolCallId: string; toolName: string };

export interface Note {
  id: string;
  docId: string;
  bbox: BBoxSelection;
  query: string;
  response: string;
  createdAt: number;
  evidence?: EvidenceWithTool[];
}

interface NotesPanelProps {
  note: Note | null;
  onClose: () => void;
  onHighlightBBox: () => void;
  onCitationClick?: (page: number, bbox_norm: [number, number, number, number]) => void;
}

// Citation preprocessing patterns
const CITE_RE = /\[page\s+(\d+)((?:,\s*span\s+[\w-]+)*)\]/gi;
const MULTI_CITE_RE =
  /\[((?:page\s+\d+(?:(?:,\s*span\s+[\w-]+)*);\s*)*page\s+\d+(?:(?:,\s*span\s+[\w-]+)*))\]/gi;
const SINGLE_ENTRY_RE = /page\s+(\d+)((?:,\s*span\s+[\w-]+)*)/gi;
const SPAN_ID_RE = /span\s+([\w-]+)/gi;

function extractSpanIds(spanSuffix: string): string[] {
  return [...spanSuffix.matchAll(SPAN_ID_RE)].map((m) => m[1]);
}

function citeBubble(page: string, spanSuffix: string): string {
  const spanIds = extractSpanIds(spanSuffix || "");
  if (spanIds.length === 0) {
    return `[cite](#__cite__${page})`;
  }
  return `[cite](#__cite__${page}__${spanIds[0]})`;
}

function preprocessCitations(content: string): string {
  let result = content.replace(MULTI_CITE_RE, (match) => {
    const entries = [...match.matchAll(SINGLE_ENTRY_RE)];
    if (entries.length <= 1) return match;
    return entries.map((e) => citeBubble(e[1], e[2] || "")).join("");
  });
  result = result.replace(CITE_RE, (_match, page, spanSuffix) => {
    return citeBubble(page, spanSuffix || "");
  });
  return result;
}

function parseCiteHref(href: string): { page: number; spanId?: string } | null {
  const match = href.match(/^#__cite__(\d+)(?:__([\w-]+))?$/);
  if (!match) return null;
  return { page: parseInt(match[1], 10), spanId: match[2] || undefined };
}

function InlineCiteBubble({
  page,
  spanId,
  evidence,
  onCitationClick,
}: {
  page: number;
  spanId?: string;
  evidence?: EvidenceWithTool[];
  onCitationClick?: (page: number, bbox_norm: [number, number, number, number]) => void;
}) {
  function handleClick() {
    if (!onCitationClick) return;
    if (evidence) {
      // try exact span_id match first, then fall back to page match
      const match = spanId
        ? evidence.find((ev) => ev.span_id === spanId)
        : evidence.find((ev) => ev.page === page);
      if (match) {
        onCitationClick(match.page, match.bbox_norm);
        return;
      }
    }
    // fallback: scroll to page with full-page highlight
    onCitationClick(page, [0, 0, 1, 1]);
  }

  return (
    <button type="button" className="cite-bubble" onClick={handleClick}>
      <BiSolidFilePdf />
      <span>p.{page + 1}</span>
    </button>
  );
}

export function NotesPanel({ note, onClose, onHighlightBBox, onCitationClick }: NotesPanelProps) {
  const [isHighlighting, setIsHighlighting] = useState(false);

  if (!note) return null;

  const handleHighlight = () => {
    setIsHighlighting(true);
    onHighlightBBox();
    setTimeout(() => setIsHighlighting(false), 200);
  };

  const createdDate = new Date(note.createdAt).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="notes-panel">
      <div className="notes-panel-header">
        <h3 className="notes-panel-title">Note</h3>
        <button
          className="notes-panel-close"
          onClick={onClose}
          aria-label="Close note"
        >
          <IoClose />
        </button>
      </div>

      <div className="notes-panel-content">
        <div className="notes-panel-section">
          <div className="notes-panel-label">Query</div>
          <div className="notes-panel-query">{note.query}</div>
        </div>

        <div className="notes-panel-section">
          <div className="notes-panel-label">Response</div>
          <div className="notes-panel-response">
            <Markdown
              remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: false }]]}
              rehypePlugins={[rehypeMathjax]}
              components={{
                a: ({ href, children, ...props }) => {
                  if (href) {
                    const cite = parseCiteHref(href);
                    if (cite) {
                      return (
                        <InlineCiteBubble
                          page={cite.page}
                          spanId={cite.spanId}
                          evidence={note.evidence}
                          onCitationClick={onCitationClick}
                        />
                      );
                    }
                  }
                  return <a href={href} {...props}>{children}</a>;
                },
              }}
            >
              {preprocessCitations(note.response)}
            </Markdown>
            </div>
        </div>

        <div className="notes-panel-meta">
          <span className="notes-panel-timestamp">{createdDate}</span>
        </div>
      </div>

      <div className="notes-panel-footer">
        <button
          className={`notes-panel-highlight-btn ${isHighlighting ? "active" : ""}`}
          onClick={handleHighlight}
        >
          Show Selection{isHighlighting ? "..." : ""}
        </button>
      </div>
    </div>
  );
}
