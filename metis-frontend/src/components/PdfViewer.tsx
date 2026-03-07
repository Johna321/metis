import { useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

const MIN_DRAG_PX = 5;

export type BBoxSelection = {
  page: number;  // 0-indexed
  bbox_norm: [number, number, number, number];  // [x0, y0, x1, y1] in [0, 1]
  bbox_pdf:  [number, number, number, number];  // [x0, y0, x1, y1] in PDF points (PyMuPDF)
};

type DragState = {
  page: number;  // 1-based page number of the active drag
  startX: number;
  startY: number;
};

type LiveRect = { left: number; top: number; width: number; height: number };

interface PageProxy {
  getViewport(opts: { scale: number }): { width: number; height: number };
}

interface PdfViewerProps {
  pdfUrl: string;
  numPages: number;
  bboxMode: boolean;
  bboxSelections: BBoxSelection[];
  onNumPagesLoad: (n: number) => void;
  onBBoxAdd: (sel: BBoxSelection) => void;
}

export function PdfViewer({ pdfUrl, numPages, bboxMode, bboxSelections, onNumPagesLoad, onBBoxAdd }: PdfViewerProps) {
  const [pageDims, setPageDims] = useState<Record<number, { w: number; h: number }>>({});
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [liveRect, setLiveRect] = useState<LiveRect | null>(null);

  function handlePageLoad(pageNum: number, page: PageProxy) {
    const { width, height } = page.getViewport({ scale: 1 });
    setPageDims(prev => ({ ...prev, [pageNum]: { w: width, h: height } }));
  }

  function handleBBoxDown(e: React.MouseEvent<HTMLDivElement>, pageNum: number) {
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    setDragState({ page: pageNum, startX: e.clientX - rect.left, startY: e.clientY - rect.top });
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

    if (dw < MIN_DRAG_PX || dh < MIN_DRAG_PX) return;

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

    // console.log("[BBox] new selection:", { page: pageNum - 1, bbox_norm: [x0n, y0n, x1n, y1n], bbox_pdf });
    onBBoxAdd({ page: pageNum - 1, bbox_norm: [x0n, y0n, x1n, y1n], bbox_pdf });
  }

  return (
    <Document
      file={pdfUrl}
      onLoadSuccess={(d) => onNumPagesLoad(d.numPages)}
      loading={<div className="status">Loading PDF...</div>}
      error={<div className="status">Failed to load PDF from backend.</div>}
    >
      {Array.from({ length: numPages }, (_, i) => (
        <div
          className={`page-wrap${bboxMode ? " no-select" : ""}`}
          key={`p_${i + 1}`}
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
  );
}
