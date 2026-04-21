import { useEffect, useRef, useState, useCallback } from "react";
import { IoArrowUp } from "react-icons/io5";
import {
  chatStart,
  listConversations,
  createConversation,
  getConversation,
  updateConversation,
  deleteConversation,
} from "../backend/http";
import type { BBoxSelection } from "./PdfViewer";
import type { ConversationMeta } from "../backend/http";
import { ChatMessageBubble, type ChatMessage } from "./ChatMessage";
import { ConversationList } from "./ConversationList";

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
  onCreateNote?: (query: string, response: string, bbox: BBoxSelection, evidence?: ChatMessage["evidence"]) => void;
}

export function ChatPanel({
  docId,
  bboxSelections,
  onBBoxClear,
  isMinimized,
  onCitationClick,
  contextText,
  onContextTextChange,
  onCreateNote,
}: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeToolCall, setActiveToolCall] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatMessagesRef = useRef<HTMLDivElement>(null);
  const unlistenRef = useRef<(() => void) | null>(null);
  const isStuckToBottom = useRef(true);

  const [conversations, setConversations] = useState<ConversationMeta[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [isConvListCollapsed, setIsConvListCollapsed] = useState(false);
  const currentQueryRef = useRef<{ query: string; bbox: BBoxSelection | null }>({ query: "", bbox: null });
  const lastNoteCreatedRef = useRef<number>(0);

  function handleMessagesScroll() {
    const el = chatMessagesRef.current;
    if (!el) return;
    // Consider "at bottom" if within 40px of the bottom
    isStuckToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }

  useEffect(() => {
    if (isStuckToBottom.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, activeToolCall]);

  useEffect(() => () => { unlistenRef.current?.(); }, []);

  // Create note when agent finishes and bbox was present
  useEffect(() => {
    if (isStreaming || !onCreateNote || !currentQueryRef.current.bbox) return;

    const lastMsg = messages[messages.length - 1];
    if (!lastMsg || lastMsg.role !== "assistant" || !lastMsg.content) return;

    // Only create once per response (use message count as a unique ID)
    const msgHash = messages.length;
    if (lastNoteCreatedRef.current === msgHash) return;

    lastNoteCreatedRef.current = msgHash;
    onCreateNote(
      currentQueryRef.current.query,
      lastMsg.content,
      currentQueryRef.current.bbox!,
      lastMsg.evidence
    );
  }, [isStreaming, messages, onCreateNote]);

  // Fetch conversation list when docId changes
  useEffect(() => {
    if (!docId) {
      setConversations([]);
      setActiveConvId(null);
      setMessages([]);
      return;
    }
    listConversations(docId).then(res => {
      setConversations(res.conversations);
    }).catch(console.error);
  }, [docId]);

  // Load messages when active conversation changes
  useEffect(() => {
    if (!docId || !activeConvId) {
      setMessages([]);
      return;
    }
    getConversation(docId, activeConvId).then(res => {
      setMessages(res.messages.map(m => ({
        role: m.role as "user" | "assistant",
        content: m.content,
        evidence: (m.evidence ?? undefined) as ChatMessage["evidence"],
      })));
    }).catch(console.error);
  }, [docId, activeConvId]);

  const refreshConversations = useCallback(() => {
    if (!docId) return;
    listConversations(docId).then(res => setConversations(res.conversations)).catch(console.error);
  }, [docId]);

  async function handleCreateConversation() {
    if (!docId) return;
    const conv = await createConversation(docId);
    setActiveConvId(conv.id);
    setMessages([]);
    refreshConversations();
  }

  async function handleRename(convId: string, title: string) {
    if (!docId) return;
    await updateConversation(docId, convId, { title });
    refreshConversations();
  }

  async function handleDelete(convId: string) {
    if (!docId) return;
    await deleteConversation(docId, convId);
    if (activeConvId === convId) {
      setActiveConvId(null);
      setMessages([]);
    }
    refreshConversations();
  }

  async function handlePin(convId: string, pinned: boolean) {
    if (!docId) return;
    await updateConversation(docId, convId, { pinned });
    refreshConversations();
  }

  async function handleSend() {
    const text = chatInput.trim();
    if (!text || !docId) return;

    unlistenRef.current?.();
    setChatInput("");
    setIsStreaming(true);
    isStuckToBottom.current = true;

    // store query and bbox for note creation
    currentQueryRef.current = {
      query: text,
      bbox: bboxSelections.length > 0 ? bboxSelections[0] : null,
    };

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
      onTitleUpdate: (_convId, _title) => {
        refreshConversations();
      },
      onAgentDone: () => {
        setActiveToolCall(null);
        setIsStreaming(false);
        unlistenRef.current?.();
        unlistenRef.current = null;
        refreshConversations();
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
    }, {
      ...(bboxSelections.length > 0 ? { selections: bboxSelections.map(s => ({ page: s.page, bbox_norm: s.bbox_norm })) } : {}),
      convId: activeConvId ?? undefined,
    });

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
          {docId && (
            <ConversationList
              conversations={conversations}
              activeConvId={activeConvId}
              isCollapsed={isConvListCollapsed}
              onSelect={setActiveConvId}
              onCreate={handleCreateConversation}
              onRename={handleRename}
              onDelete={handleDelete}
              onPin={handlePin}
              onToggleCollapse={() => setIsConvListCollapsed(!isConvListCollapsed)}
            />
          )}

          {!docId ? (
            <div className="panel-empty">Open a PDF to start chatting.</div>
          ) : !activeConvId ? (
            <div className="conv-empty-state">
              <div className="conv-empty-state-icon">&#128172;</div>
              <div className="conv-empty-state-title">No conversation selected</div>
              <div className="conv-empty-state-subtitle">Start a conversation to ask questions about this paper</div>
              <button className="conv-empty-state-btn" onClick={handleCreateConversation}>New Conversation</button>
            </div>
          ) : (
            <>
              <div className="chat-messages" ref={chatMessagesRef} onScroll={handleMessagesScroll}>
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
                      x
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
                    disabled={!docId || !activeConvId || isStreaming}
                  />
                  <button
                    className="chat-send-btn"
                    onClick={handleSend}
                    disabled={!docId || !activeConvId || !chatInput.trim() || isStreaming}
                  >
                    <IoArrowUp />
                  </button>
                </div>
              </div>
            </>
          )}
        </>
      )}
    </aside>
  );
}
