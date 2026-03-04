import { useEffect, useRef, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";

import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";

import "./App.css";

import { ingestPdf, retrieveEvidence, getDocumentPdfUrl, chatStart, type EvidenceItem, type BboxSelection } from "./backend/http";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

const MIN_DRAG_PX = 5;

// BBox types

type BBoxSelection = {
  page: number;  // 0-indexed
  bbox_norm: [number, number, number, number];  // [x0, y0, x1, y1] in [0, 1]
  bbox_pdf:  [number, number, number, number];  // [x0, y0, x1, y1] in PDF points (PyMuPDF)
};

type DragState = {
  page: number;  // 1-based page number of the active drag
  startX: number;  // px relative to overlay
  startY: number;
};

type LiveRect = { left: number; top: number; width: number; height: number };

interface PageProxy {
  getViewport(opts: { scale: number }): { width: number; height: number };
}

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

function App() {
  const [docId, setDocId] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number>(0);

  const [evidence, setEvidence] = useState<EvidenceItem[]>([]);
  const [status, setStatus] = useState<string>("");

  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const unlistenRef = useRef<(() => void) | null>(null);

  const [pdfUrl, setPdfUrl] = useState<string | null>(null);

  // BBox state
  const [bboxMode, setBboxMode] = useState(false);
  const [pageDims, setPageDims] = useState<Record<number, { w: number; h: number }>>({});
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [liveRect, setLiveRect] = useState<LiveRect | null>(null);
  const [bboxSelections, setBboxSelections] = useState<BBoxSelection[]>([]);

  // Resolve PDF URL whenever docId changes
  useEffect(() => {
    if (!docId) {
      setPdfUrl(null);
      return;
    }
    getDocumentPdfUrl(docId).then(setPdfUrl);
  }, [docId]);

  async function openPdf() {
    const path = await open({
      multiple: false,
      filters: [{ name: "PDF", extensions: ["pdf"] }],
    });

    if (!path || Array.isArray(path)) return;

    setStatus("Ingesting...");
    setEvidence([]);
    setDocId(null);
    setNumPages(0);

    try {
      const meta = await ingestPdf(path);
      setDocId(meta.doc_id);
      setStatus("");
    } catch (err: unknown) {
      console.error("ingest_pdf error:", err);
      setStatus(`Ingest failed: ${err}`);
    }
  }

  // capture text selection & send retrieve request
  async function handleMouseUp(pageNumberOneBased: number) {
    if (!docId) return;

    const selected = window.getSelection()?.toString().trim() ?? "";
    if (selected.length < 3) return; // ignore tiny selections

    // setStatus("Retrieving evidence...");
    // try {
    //   const ev = await retrieveEvidence(docId, pageNumberOneBased - 1, selected);
    //   setEvidence(ev);
    //   setStatus("");
    // } catch (err) {
    //   console.error(err);
    //   setStatus("Retrieve failed.");
    // }
  }

  // BBox handlers

  function handlePageLoad(pageNum: number, page: PageProxy) {
    const { width, height } = page.getViewport({ scale: 1 });
    setPageDims(prev => ({ ...prev, [pageNum]: { w: width, h: height } }));
  }

  function handleBBoxToggle() {
    if (bboxMode) {  // confirm
      // log accumulated selections and exit mode
      if (bboxSelections.length > 0) {
        console.log("[BBox] selections confirmed:", bboxSelections);
        // todo: wire into backend
      }
      setBboxMode(false);
      setBboxSelections([]);
      setDragState(null);
      setLiveRect(null);
    } else {
      setBboxMode(true);
      setBboxSelections([]);
    }
  }

  function handleBBoxDown(e: React.MouseEvent<HTMLDivElement>, pageNum: number) {
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    setDragState({
      page: pageNum,
      startX: e.clientX - rect.left,
      startY: e.clientY - rect.top,
    });
    setLiveRect(null);
  }

  function handleBBoxMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!dragState) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const curX = e.clientX - rect.left;
    const curY = e.clientY - rect.top;
    setLiveRect({
      left:   Math.min(dragState.startX, curX),
      top:    Math.min(dragState.startY, curY),
      width:  Math.abs(curX - dragState.startX),
      height: Math.abs(curY - dragState.startY),
    });
  }

  function handleBBoxUp(e: React.MouseEvent<HTMLDivElement>, pageNum: number) {
    if (!dragState || dragState.page !== pageNum) return;

    const overlay = e.currentTarget;
    const renderedW = overlay.offsetWidth;
    const renderedH = overlay.offsetHeight;
    const rect = overlay.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;

    const dw = Math.abs(endX - dragState.startX);
    const dh = Math.abs(endY - dragState.startY);

    setDragState(null);
    setLiveRect(null);

    if (dw < MIN_DRAG_PX || dh < MIN_DRAG_PX) return;  // ignore accidental clicks

    const x0 = Math.min(dragState.startX, endX);
    const y0 = Math.min(dragState.startY, endY);
    const x1 = Math.max(dragState.startX, endX);
    const y1 = Math.max(dragState.startY, endY);

    const x0n = x0 / renderedW;
    const y0n = y0 / renderedH;
    const x1n = x1 / renderedW;
    const y1n = y1 / renderedH;

    const dims = pageDims[pageNum];
    const bbox_pdf: [number, number, number, number] = dims
      ? [x0n * dims.w, y0n * dims.h, x1n * dims.w, y1n * dims.h]
      : [0, 0, 0, 0];

    const selection: BBoxSelection = {
      page: pageNum - 1,  // convert to 0-indexed
      bbox_norm: [x0n, y0n, x1n, y1n],
      bbox_pdf,
    };

    console.log("[BBox] new selection:", selection);
    setBboxSelections(prev => [...prev, selection]);
  }

  // scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Clean up stream listener on unmount
  useEffect(() => () => { unlistenRef.current?.(); }, []);

  async function handleSend() {
    const text = chatInput.trim();
    if (!text || !docId) return;

    unlistenRef.current?.();
    setChatInput("");
    setIsStreaming(true);
    setMessages(prev => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "" },
    ]);

    unlistenRef.current = await chatStart(docId, text, {
      onTextDelta: (delta) => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          updated[updated.length - 1] = { ...last, content: last.content + delta };
          return updated;
        });
        console.log(delta);
      },
      onMessageDone: (_role, _content, toolCalls) => {
        // Only clean up when the final message has no tool calls.
        // Intermediate message_done events (with tool_calls) mean the agent
        // is still running another iteration after executing the tools.
        if (!toolCalls || (Array.isArray(toolCalls) && toolCalls.length === 0)) {
          setIsStreaming(false);
          unlistenRef.current?.();
          unlistenRef.current = null;
        }
      },
      onError: (msg) => {
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

    // Clear bbox selections after sending message
    setBboxSelections([]);
  }

  function handleChatKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // ---

  return (
    <div className="app-shell">
      <header className="toolbar">
        <button onClick={openPdf}>Open PDF</button>

        <button
          className={`bbox-btn${bboxMode ? " active" : ""}`}
          onClick={handleBBoxToggle}
          disabled={!docId}
        >
          BBox
        </button>

        <div className="spacer" />
        <span>{status}</span>
        <span style={{ marginLeft: 12 }}>{numPages ? `${numPages} pages` : ""}</span>
      </header>

      <div className="main-split">
        <main className="viewer">
          {pdfUrl ? (
            <Document
              file={pdfUrl}
              onLoadSuccess={(d) => setNumPages(d.numPages)}
              loading={<div className="status">Loading PDF...</div>}
              error={<div className="status">Failed to load PDF from backend.</div>}
            >
              {Array.from({ length: numPages }, (_, i) => (
                <div
                  className={`page-wrap${bboxMode ? " no-select" : ""}`}
                  key={`p_${i + 1}`}
                  onMouseUp={bboxMode ? undefined : () => handleMouseUp(i + 1)}
                >
                  <Page
                    pageNumber={i + 1}
                    width={900}
                    onLoadSuccess={(page) => handlePageLoad(i + 1, page as unknown as PageProxy)}
                  />
                  {bboxMode && (
                    <div
                      className="bbox-overlay"
                      onMouseDown={(e) => handleBBoxDown(e, i + 1)}
                      onMouseMove={handleBBoxMove}
                      onMouseUp={(e) => handleBBoxUp(e, i + 1)}
                    >
                      {bboxSelections
                        .filter(b => b.page === i)
                        .map((b, idx) => (
                          <div
                            key={idx}
                            className="bbox-completed"
                            style={{
                              left:   `${b.bbox_norm[0] * 100}%`,
                              top:    `${b.bbox_norm[1] * 100}%`,
                              width:  `${(b.bbox_norm[2] - b.bbox_norm[0]) * 100}%`,
                              height: `${(b.bbox_norm[3] - b.bbox_norm[1]) * 100}%`,
                            }}
                          />
                        ))}
                      {liveRect && dragState?.page === i + 1 && (
                        <div className="bbox-rubber-band" style={liveRect} />
                      )}
                    </div>
                  )}
                </div>
              ))}
            </Document>
          ) : (
            <div className="status">Open a PDF to begin</div>
          )}
        </main>

        <aside className="sidepanel">
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="panel-empty">Ask a question about the document.</div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`chat-bubble chat-bubble--${msg.role}`}>
                {msg.content}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className="chat-input-row">
            <textarea
              className="chat-textarea"
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={handleChatKeyDown}
              placeholder="Ask about the paper… (Enter to send)"
              rows={3}
              disabled={!docId || isStreaming}
            />
            <button
              className="chat-send-btn"
              onClick={handleSend}
              disabled={!docId || !chatInput.trim() || isStreaming}
            >
              Send
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
