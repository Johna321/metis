import { useMemo, useState } from "react";
import reactLogo from "./assets/react.svg";
import { invoke } from "@tauri-apps/api/core";
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
  const [page, setPage] = useState<number>(1);

  async function openPdf() {
    const path = await open({
      multiple: false,
      filters: [{ name: "PDF", extensions: ["pdf"] }],
    });

    if (!path || Array.isArray(path)) return;

    const bytes = await readFile(path);
    setPdfData(bytes);
    setPage(1);
  }

  const file = useMemo(() => (pdfData ? { data: pdfData } : null), [pdfData]);

  return (
    <div className="app-shell">
      <header className="toolbar">
        <button onClick={openPdf}>Open PDF</button>
        <div className="spacer" />
        <span>{numPages ? `${numPages} pages` : ""}</span>
      </header>

      <main className="viewer">
        {file ? (
          <Document
            file={file}
            onLoadSuccess={(d) => setNumPages(d.numPages)}
            loading={<div className="status">Loading…</div>}
            error={<div className="status">Failed to load PDF.</div>}
          >
            {Array.from({ length: numPages }, (_, i) => (
              <div className="page-wrap" key={`p_${i + 1}`}>
                <Page
                  pageNumber={i + 1}
                  // You can start with a fixed width; later we can make this responsive.
                  width={900}
                />
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
