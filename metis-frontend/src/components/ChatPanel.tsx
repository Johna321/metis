import { useEffect, useRef, useState } from "react";
import { IoArrowUp } from "react-icons/io5";
import { chatStart } from "../backend/http";
import type { BBoxSelection } from "./PdfViewer";
import { ChatMessageBubble, type ChatMessage } from "./ChatMessage";

const TOOL_CALL_GENERIC: Record<string, string> = {
  rag_retrieve: "Searching paper",
  read_page: "Reading page",
  web_search: "Searching the web",
};

function formatToolCall(name: string, args: unknown): string {
  const a = args as Record<string, unknown> | null;
  switch (name) {
    case "rag_retrieve":
      return a?.query ? `Searching paper for "${a.query}"` : "Searching paper";
    case "read_page":
      return a?.page != null ? `Reading page ${(a.page as number) + 1}` : "Reading page";
    case "web_search":
      return a?.query ? `Searching the web for "${a.query}"` : "Searching the web";
    default:
      return name;
  }
}

interface ChatPanelProps {
  docId: string | null;
  bboxSelections: BBoxSelection[];
  onBBoxClear: () => void;
  isMinimized: boolean;
  onCitationClick: (page: number, bbox_norm: [number, number, number, number]) => void;
  contextText?: string | null;
  onContextTextChange?: (text: string | null) => void;
}

export function ChatPanel({
  docId,
  bboxSelections,
  onBBoxClear,
  isMinimized,
  onCitationClick,
  contextText,
  onContextTextChange,
}: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeToolCall, setActiveToolCall] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const unlistenRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => () => { unlistenRef.current?.(); }, []);

  async function handleSend() {
    const text = chatInput.trim();
    if (!text || !docId) return;

    unlistenRef.current?.();
    setChatInput("");
    setIsStreaming(true);

    // prepend context if available
    let messageWithContext = text;
    if (contextText) {
      messageWithContext = `Based on the following excerpt from the PDF:\n\n"${contextText}"\n\nMy question: ${text}`;
      onContextTextChange?.(null);
    }

    setMessages(prev => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "" },
    ]);

    unlistenRef.current = await chatStart(docId, messageWithContext, {
      onToolCallStart: (name) => {
        setActiveToolCall(TOOL_CALL_GENERIC[name] ?? name);
      },
      onToolCallDone: (_id, name, args) => {
        setActiveToolCall(formatToolCall(name, args));
      },
      onTextDelta: (delta) => {
        setActiveToolCall(null);
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          updated[updated.length - 1] = { ...last, content: last.content + delta };
          return updated;
        });
      },
      onCitationData: (items, toolCallId, toolName) => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          const tagged = items.map(item => ({ ...item, toolCallId, toolName }));
          const merged = [...(last.evidence ?? []), ...tagged];
          updated[updated.length - 1] = { ...last, evidence: merged };
          return updated;
        });
      },
      onAgentDone: () => {
        setActiveToolCall(null);
        setIsStreaming(false);
        unlistenRef.current?.();
        unlistenRef.current = null;
      },
      onError: (msg) => {
        setActiveToolCall(null);
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: `Error: ${msg}` };
          return updated;
        });
        setIsStreaming(false);
        unlistenRef.current?.();
        unlistenRef.current = null;
      },
    }, bboxSelections.length > 0
      ? { selections: bboxSelections.map(s => ({ page: s.page, bbox_norm: s.bbox_norm })) }
      : undefined,
    );

    onBBoxClear();
  }

  function handleChatKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <aside className={`sidepanel ${isMinimized ? "minimized" : ""}`}>
      {!isMinimized && (
        <>
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="panel-empty">Ask a question about the document.</div>
            )}
            {messages.map((msg, i) => (
              <ChatMessageBubble
                key={i}
                msg={msg}
                isPending={isStreaming && i === messages.length - 1}
                isStreaming={isStreaming && i === messages.length - 1}
                onCitationClick={onCitationClick}
              />
            ))}
            {activeToolCall && (
              <div className="tool-call-indicator">{activeToolCall}</div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="chat-input-row">
            {contextText && (
              <div className="context-badge">
                <span className="context-text" title={"(" + `${contextText.length}` + " chars)"}>
                  Context added: {contextText.substring(0, 30)}{contextText.length >= 30 && "..."}
                </span>
                <button
                  className="context-clear"
                  onClick={() => onContextTextChange?.(null)}
                  title="Remove context"
                >
                  ×
                </button>
              </div>
            )}
            <div className="chat-input-wrap">
              <textarea
                className="chat-textarea"
                value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                onKeyDown={handleChatKeyDown}
                placeholder="Ask about the paper... (Enter to send)"
                rows={3}
                disabled={!docId || isStreaming}
              />
              <button
                className="chat-send-btn"
                onClick={handleSend}
                disabled={!docId || !chatInput.trim() || isStreaming}
              >
                <IoArrowUp />
              </button>
            </div>
          </div>
        </>
      )}
    </aside>
  );
}
