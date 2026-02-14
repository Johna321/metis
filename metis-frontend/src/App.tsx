import { useMemo, useRef, useState } from "react";
import { isTauri } from '@tauri-apps/api/core';
import { open } from "@tauri-apps/plugin-dialog";
import { readFile } from "@tauri-apps/plugin-fs";

import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";

import "./App.css";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();


function App() {
  const [pdfData, setPdfData] = useState<Uint8Array | null>(null);
  const [numPages, setNumPages] = useState<number>(0);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  async function openPdf() {
    if (isTauri()) {
      const path = await open({
        multiple: false,
        filters: [{ name: "PDF", extensions: ["pdf"] }],
      });

      if (!path || Array.isArray(path)) return;

      const bytes = await readFile(path);
      setPdfData(bytes);
      setNumPages(0);
      return;
    }

    // web
    fileInputRef.current?.click();
  }

  async function onWebFilePicked(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    const buf = await f.arrayBuffer();
    setPdfData(new Uint8Array(buf));
    setNumPages(0);

    e.target.value = "";  // allow picking same file twice
  }

  const file = useMemo(() => (pdfData ? { data: pdfData } : null), [pdfData]);

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
        <span>{numPages ? `${numPages} pages` : ""}</span>
      </header>

      <main className="viewer">
        {file ? (
          <Document file={file} onLoadSuccess={(d) => setNumPages(d.numPages)}>
            {Array.from({ length: numPages }, (_, i) => (
              <div className="page-wrap" key={`p_${i + 1}`}>
                <Page pageNumber={i + 1} width={900} />
              </div>
            ))}
          </Document>
        ) : (
          <div className="status">Open a PDF to begin</div>
        )}
      </main>
    </div>
  );
}

export default App;
