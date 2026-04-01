import { useState } from "react";
import Markdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeMathjax from "rehype-mathjax";
import { DotPulse } from "ldrs/react";
import "ldrs/react/DotPulse.css";
import { BiSolidFilePdf } from "react-icons/bi";
import { IoChevronDown, IoChevronUp } from "react-icons/io5";
import type { EvidenceItem } from "../backend/http";

export const TOOL_BADGE: Record<string, { label: string; cls: string }> = {
  rag_retrieve: { label: "RAG", cls: "rag" },
  web_search: { label: "WEB", cls: "web" },
  read_page: { label: "PG", cls: "page" },
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  evidence?: Array<EvidenceItem & { toolCallId: string; toolName: string }>;
};

type EvidenceWithTool = EvidenceItem & { toolCallId: string; toolName: string };

// ---------------------------------------------------------------------------
// Citation preprocessing: [page X, span Y] → markdown link [cite](#__cite__X__Y)
// ---------------------------------------------------------------------------

const CITE_RE = /\[page\s+(\d+)(?:,\s*span\s+([\w-]+))?\]/gi;

function preprocessCitations(content: string): string {
  return content.replace(CITE_RE, (_match, page, spanId) =>
    spanId
      ? `[cite](#__cite__${page}__${spanId})`
      : `[cite](#__cite__${page})`,
  );
}

function parseCiteHref(href: string): { page: number; spanId?: string } | null {
  const match = href.match(/^#__cite__(\d+)(?:__([\w-]+))?$/);
  if (!match) return null;
  return { page: parseInt(match[1], 10), spanId: match[2] || undefined };
}

// ---------------------------------------------------------------------------
// InlineCiteBubble — clickable pill rendered inside markdown text
// ---------------------------------------------------------------------------

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
      // Try exact span_id match first, then fall back to page match
      const match = spanId
        ? evidence.find((ev) => ev.span_id === spanId)
        : evidence.find((ev) => ev.page === page);
      if (match) {
        onCitationClick(match.page, match.bbox_norm);
        return;
      }
    }
    // Fallback: scroll to page with full-page highlight
    onCitationClick(page, [0, 0, 1, 1]);
  }

  return (
    <button type="button" className="cite-bubble" onClick={handleClick}>
      <BiSolidFilePdf />
      <span>p.{page + 1}</span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// SourcesSection — collapsible "Sources" button + categorized dropdown
// ---------------------------------------------------------------------------

function categorizeEvidence(
  content: string,
  evidence: EvidenceWithTool[],
): {
  citations: Array<{ page: number; spanId?: string; ev?: EvidenceWithTool }>;
  remaining: EvidenceWithTool[];
} {
  const matches = [...content.matchAll(CITE_RE)];
  const citedSpanIds = new Set<string>();
  const pageOnlyCitations: number[] = [];

  for (const m of matches) {
    const pg = parseInt(m[1], 10);
    const sid = m[2];
    if (sid) citedSpanIds.add(sid);
    else pageOnlyCitations.push(pg);
  }

  const citations: Array<{ page: number; spanId?: string; ev?: EvidenceWithTool }> = [];
  const remaining: EvidenceWithTool[] = [];

  for (const ev of evidence) {
    if (citedSpanIds.has(ev.span_id)) {
      citations.push({ page: ev.page, spanId: ev.span_id, ev });
    } else {
      remaining.push(ev);
    }
  }

  for (const pg of pageOnlyCitations) {
    if (!citations.some((c) => c.page === pg)) {
      // Try to match a remaining evidence item on this page
      const idx = remaining.findIndex((ev) => ev.page === pg);
      if (idx !== -1) {
        const [matched] = remaining.splice(idx, 1);
        citations.push({ page: pg, ev: matched });
      } else {
        citations.push({ page: pg });
      }
    }
  }

  return { citations, remaining };
}

function SourcesSection({
  evidence,
  content,
  onCitationClick,
}: {
  evidence: EvidenceWithTool[];
  content: string;
  onCitationClick?: (page: number, bbox_norm: [number, number, number, number]) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const { citations, remaining } = categorizeEvidence(content, evidence);
  const totalCount = citations.length + remaining.length;

  if (totalCount === 0) return null;

  return (
    <>
      <button
        type="button"
        className="sources-toggle"
        onClick={() => setIsOpen((v) => !v)}
      >
        <BiSolidFilePdf />
        <span>Sources</span>
        <span className="sources-count">{totalCount}</span>
        {isOpen ? <IoChevronUp /> : <IoChevronDown />}
      </button>

      {isOpen && (
        <div className="sources-dropdown">
          {citations.length > 0 && (
            <>
              <div className="sources-section-header">Cited in response</div>
              <div className="citation-list">
                {citations.map((c, j) => (
                  <div
                    key={`cite-${j}`}
                    className="citation-item"
                    title={c.ev?.text}
                    onClick={() =>
                      onCitationClick?.(
                        c.ev?.page ?? c.page,
                        c.ev?.bbox_norm ?? [0, 0, 1, 1],
                      )
                    }
                  >
                    <span className={`citation-src citation-src--${c.ev ? (TOOL_BADGE[c.ev.toolName]?.cls ?? "default") : "page"}`}>
                      {c.ev ? (TOOL_BADGE[c.ev.toolName]?.label ?? "?") : "PG"}
                    </span>
                    <span className="citation-page">p.{(c.ev?.page ?? c.page) + 1}</span>
                    {c.ev && (
                      <>
                        <span className="citation-score">{(c.ev.score * 100).toFixed(0)}%</span>
                        <span className="citation-text">
                          {c.ev.text.slice(0, 80)}{c.ev.text.length > 80 ? "..." : ""}
                        </span>
                      </>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}

          {remaining.length > 0 && (
            <>
              <div className="sources-section-header">Retrieved evidence</div>
              <div className="citation-list">
                {remaining.map((ev, j) => (
                  <div
                    key={`ev-${j}`}
                    className="citation-item"
                    title={ev.text}
                    onClick={() => onCitationClick?.(ev.page, ev.bbox_norm)}
                  >
                    <span className={`citation-src citation-src--${TOOL_BADGE[ev.toolName]?.cls ?? "default"}`}>
                      {TOOL_BADGE[ev.toolName]?.label ?? "?"}
                    </span>
                    <span className="citation-page">p.{ev.page + 1}</span>
                    <span className="citation-score">{(ev.score * 100).toFixed(0)}%</span>
                    <span className="citation-text">
                      {ev.text.slice(0, 80)}{ev.text.length > 80 ? "..." : ""}
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// ChatMessageBubble — main export
// ---------------------------------------------------------------------------

interface ChatMessageBubbleProps {
  msg: ChatMessage;
  isPending?: boolean;
  isStreaming?: boolean;
  onCitationClick?: (page: number, bbox_norm: [number, number, number, number]) => void;
}

export function ChatMessageBubble({ msg, isPending, isStreaming, onCitationClick }: ChatMessageBubbleProps) {
  const processedContent = isStreaming ? msg.content : preprocessCitations(msg.content);

  return (
    <div className={`chat-bubble chat-bubble--${msg.role}`}>
      {isPending && msg.content === "" ? (
        <DotPulse size={28} speed={1.3} color="currentColor" />
      ) : (
        <Markdown
          remarkPlugins={[remarkMath]}
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
                      evidence={msg.evidence}
                      onCitationClick={onCitationClick}
                    />
                  );
                }
              }
              return <a href={href} {...props}>{children}</a>;
            },
          }}
        >
          {processedContent}
        </Markdown>
      )}

      {!isStreaming && msg.evidence && msg.evidence.length > 0 && (
        <SourcesSection
          evidence={msg.evidence}
          content={msg.content}
          onCitationClick={onCitationClick}
        />
      )}
    </div>
  );
}
