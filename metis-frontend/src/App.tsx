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

function App() {
  const [docId, setDocId] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number>(0);

  const [evidence, setEvidence] = useState<EvidenceItem[]>([]);
  const [status, setStatus] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

  return (
    <div className="app-shell">
      <header className="toolbar">
        <button onClick={openPdf}>Open PDF</button>

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
                  className="page-wrap"
                  key={`p_${i + 1}`}
                  onMouseUp={() => handleMouseUp(i + 1)}
                >
                  <Page pageNumber={i + 1} width={900} />
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
