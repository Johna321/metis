import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { convertFileSrc } from "@tauri-apps/api/core";

import "./App.css";

import { ingestPdf, vectorizeDoc } from "./backend/http";
import { Toolbar } from "./components/Toolbar";
import { PdfViewer, type BBoxSelection } from "./components/PdfViewer";
import { ChatPanel } from "./components/ChatPanel";

function App() {
  const [docId, setDocId] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number>(0);
  const [status, setStatus] = useState<string>("");
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [bboxMode, setBboxMode] = useState(false);
  const [bboxSelections, setBboxSelections] = useState<BBoxSelection[]>([]);

  async function openPdf() {
    const path = await open({
      multiple: false,
      filters: [{ name: "PDF", extensions: ["pdf"] }],
    });

    if (!path || Array.isArray(path)) return;

    setDocId(null);
    setNumPages(0);
    setPdfUrl(convertFileSrc(path));
    setStatus("Ingesting...");

    try {
      const meta = await ingestPdf(path);
      setStatus("Vectorizing...");
      const vec = await vectorizeDoc(meta.doc_id);
      setDocId(meta.doc_id);
      setStatus(vec.was_cached ? "Loaded (embeddings cached)" : "");
      if (vec.was_cached) setTimeout(() => setStatus(""), 2000);
    } catch (err: unknown) {
      console.error("ingest error:", err);
      setStatus(`Ingest failed: ${err}`);
    }
  }

  function handleBBoxToggle() {
    if (bboxMode) {
      if (bboxSelections.length > 0) {
        console.log("[BBox] selections confirmed:", bboxSelections);
        // todo: wire into backend
      }
      setBboxMode(false);
      setBboxSelections([]);
    } else {
      setBboxMode(true);
      setBboxSelections([]);
    }
  }

  return (
    <div className="app-shell">
      <Toolbar
        onOpenPdf={openPdf}
        onBBoxToggle={handleBBoxToggle}
        bboxMode={bboxMode}
        docId={docId}
        status={status}
        numPages={numPages}
      />

      <div className="main-split">
        <main className="viewer">
          {pdfUrl ? (
            <PdfViewer
              pdfUrl={pdfUrl}
              numPages={numPages}
              bboxMode={bboxMode}
              bboxSelections={bboxSelections}
              onNumPagesLoad={setNumPages}
              onBBoxAdd={(sel) => setBboxSelections(prev => [...prev, sel])}
            />
          ) : (
            <div className="status">Open a PDF to begin</div>
          )}
        </main>

        <ChatPanel
          docId={docId}
          bboxSelections={bboxSelections}
          onBBoxClear={() => setBboxSelections([])}
        />
      </div>
    </div>
  );
}

export default App;
