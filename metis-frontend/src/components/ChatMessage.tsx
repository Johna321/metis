import Markdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeMathjax from 'rehype-mathjax'
import { DotPulse } from 'ldrs/react'
import 'ldrs/react/DotPulse.css'
import type { EvidenceItem } from "../backend/http";

export const TOOL_BADGE: Record<string, { label: string; cls: string }> = {
  rag_retrieve: { label: "RAG", cls: "rag" },
  web_search:   { label: "WEB", cls: "web" },
  read_page:    { label: "PG",  cls: "page" },
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  evidence?: Array<EvidenceItem & { toolCallId: string; toolName: string }>;
};

interface ChatMessageBubbleProps {
  msg: ChatMessage;
  isPending?: boolean;
  onCitationClick?: (page: number, bbox_norm: [number, number, number, number]) => void;
}

export function ChatMessageBubble({ msg, isPending, onCitationClick }: ChatMessageBubbleProps) {
  return (
    <div className={`chat-bubble chat-bubble--${msg.role}`}>
      {isPending && msg.content === ""
        ? <DotPulse size={28} speed={1.3} color="currentColor" />
        : <Markdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeMathjax]}>{msg.content}</Markdown>
      }
      {msg.evidence && msg.evidence.length > 0 && (
        <div className="citation-list">
          {msg.evidence.map((ev, j) => (
            <div key={j} className="citation-item" title={ev.text} onClick={() => onCitationClick?.(ev.page, ev.bbox_norm)} style={{ cursor: "pointer" }}>
              <span className={`citation-src citation-src--${TOOL_BADGE[ev.toolName]?.cls ?? "default"}`}>
                {TOOL_BADGE[ev.toolName]?.label ?? "?"}
              </span>
              <span className="citation-page">p.{ev.page + 1}</span>
              <span className="citation-score">{(ev.score * 100).toFixed(0)}%</span>
              <span className="citation-text">{ev.text.slice(0, 80)}{ev.text.length > 80 ? "..." : ""}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
