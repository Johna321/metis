import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { convertFileSrc } from "@tauri-apps/api/core";

import "./App.css";

import { hashFile, checkDoc, ingestPdf, vectorizeDoc } from "./backend/http";
import { Toolbar } from "./components/Toolbar";
import { PdfViewer, type BBoxSelection } from "./components/PdfViewer";
import { ChatPanel } from "./components/ChatPanel";
import { Settings } from "./components/Settings";

function App() {
  const [docId, setDocId] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number>(0);
  const [status, setStatus] = useState<string>("");
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [bboxMode, setBboxMode] = useState(false);
  const [bboxSelections, setBboxSelections] = useState<BBoxSelection[]>([]);
  const [isChatMinimized, setIsChatMinimized] = useState(false);
  const [highlightTarget, setHighlightTarget] = useState<{ page: number; bbox_norm: [number, number, number, number] } | null>(null);  // scroll to evidence item
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [contextText, setContextText] = useState<string | null>(null);

  async function openPdf() {
    const path = await open({
      multiple: false,
      filters: [{ name: "PDF", extensions: ["pdf"] }],
    });

    if (!path || Array.isArray(path)) return;

    setDocId(null);
    setNumPages(0);
    setPdfUrl(convertFileSrc(path));
    setStatus("Hashing...");

    try {
      const docId = await hashFile(path);

      setStatus("Checking cache...");
      let meta = await checkDoc(docId);
      const ingestCached = meta !== null;

      if (!meta) {
        setStatus("Ingesting...");
        meta = await ingestPdf(path);
      }

      setStatus("Vectorizing...");
      const vec = await vectorizeDoc(meta.doc_id);
      setDocId(meta.doc_id);

      const anyCached = ingestCached || vec.was_cached;
      setStatus(anyCached ? "Loaded (cached)" : "");
      if (anyCached) setTimeout(() => setStatus(""), 2000);
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
        isChatMinimized={isChatMinimized}
        onToggleChatMinimize={() => setIsChatMinimized(!isChatMinimized)}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      <div className="main-split">
        <main className="viewer">
          {pdfUrl ? (
            <PdfViewer
              key={pdfUrl}
              pdfUrl={pdfUrl}
              numPages={numPages}
              bboxMode={bboxMode}
              bboxSelections={bboxSelections}
              highlightTarget={highlightTarget}
              onNumPagesLoad={setNumPages}
              onBBoxAdd={(sel) => setBboxSelections(prev => [...prev, sel])}
              onBackgroundClick={() => setHighlightTarget(null)}
              onContextTextChange={setContextText}
            />
          ) : (
            <div className="status">Open a PDF to begin</div>
          )}
        </main>

        <ChatPanel
          docId={docId}
          bboxSelections={bboxSelections}
          onBBoxClear={() => setBboxSelections([])}
          isMinimized={isChatMinimized}
          onCitationClick={(page, bbox) => setHighlightTarget({ page, bbox_norm: bbox })}
          contextText={contextText}
          onContextTextChange={setContextText}
        />
      </div>

      <Settings isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
    </div>
  );
}

export default App;
