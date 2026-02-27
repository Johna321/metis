import { useMemo, useRef, useState } from "react";
import { isTauri } from '@tauri-apps/api/core';
import { open } from "@tauri-apps/plugin-dialog";
import { readFile } from "@tauri-apps/plugin-fs";

import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";

import "./App.css";

import { ingestPdf, retrieveEvidence, documentPdfUrl, type EvidenceItem } from "./backend/http";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

const API_BASE = "http://127.0.0.1:8000";
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

function App() {
  const [docId, setDocId] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number>(0);

  const [evidence, setEvidence] = useState<EvidenceItem[]>([]);
  const [status, setStatus] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // BBox state
  const [bboxMode, setBboxMode] = useState(false);
  const [pageDims, setPageDims] = useState<Record<number, { w: number; h: number }>>({});
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [liveRect, setLiveRect] = useState<LiveRect | null>(null);
  const [bboxSelections, setBboxSelections] = useState<BBoxSelection[]>([]);

  async function ingestFromFile(file: File) {
    setStatus("Ingesting...");
    setEvidence([]);
    setDocId(null);
    setNumPages(0);

    try {
      const meta = await ingestPdf(API_BASE, file);
      setDocId(meta.doc_id);
      setStatus("");
    } catch (err) {
      console.error(err);
      setStatus("Ingest failed. Is the backend running?");
    }
  }

  async function openPdf() {
    if (isTauri()) {
      const path = await open({
        multiple: false,
        filters: [{ name: "PDF", extensions: ["pdf"] }],
      });

      if (!path || Array.isArray(path)) return;

      const bytes = await readFile(path);
      const blob = new Blob([bytes], { type: "application/pdf" });
      const file = new File([blob], "document.pdf", { type: "application/pdf" });
      await ingestFromFile(file);
      return;
    }

    // web
    fileInputRef.current?.click();
  }

  async function onWebFilePicked(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    await ingestFromFile(f);
    e.target.value = "";  // allow picking same file twice
  }

  // render PDF by URL served from backend
  const pdfUrl = useMemo(() => (docId ? documentPdfUrl(API_BASE, docId) : null), [docId]);

  // capture text selection & send retrieve request
  async function handleMouseUp(pageNumberOneBased: number) {
    if (!docId) return;

    const selected = window.getSelection()?.toString().trim() ?? "";
    if (selected.length < 3) return; // ignore tiny selections

    setStatus("Retrieving evidence...");
    try {
      const ev = await retrieveEvidence(API_BASE, docId, pageNumberOneBased - 1, selected);
      setEvidence(ev);
      setStatus("");
    } catch (err) {
      console.error(err);
      setStatus("Retrieve failed.");
    }
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

        {!isTauri() && (
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            style={{ display: "none" }}
            onChange={onWebFilePicked}
          />
        )}

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
          <div className="panel-title">Evidence</div>
          {evidence.length === 0 ? (
            <div className="panel-empty">Select text in the PDF to retrieve evidence.</div>
          ) : (
            <ul className="ev-list">
              {evidence.map((e) => (
                <li key={e.span_id} className="ev-item">
                  <div className="ev-meta">
                    <span>p{e.page + 1}</span>
                    <span>{e.score.toFixed(3)}</span>
                  </div>
                  <div className="ev-text">{e.text}</div>
                </li>
              ))}
            </ul>
          )}
        </aside>
      </div>
    </div>
  );
}

export default App;
