import { useRef, useEffect, useState, useCallback } from "react";

import "pdfjs-dist/web/pdf_viewer.css";

import { usePdfDocument } from "../hooks/usePdfDocument";
import { useVirtualPages } from "../hooks/useVirtualPages";
import { useRenderQueue } from "../hooks/useRenderQueue";
import { usePageLayers } from "../hooks/usePageLayers";
import { useBBoxDrag } from "../hooks/useBBoxDrag";
import { PageSlot } from "./PageSlot";
import { BBoxOverlay } from "./BBoxOverlay";

const PAGE_WIDTH = 900;

export type BBoxSelection = {
  page: number; // 0-indexed
  bbox_norm: [number, number, number, number]; // [x0, y0, x1, y1] in [0, 1]
  bbox_pdf: [number, number, number, number]; // [x0, y0, x1, y1] in PDF points (PyMuPDF)
};

interface PdfViewerProps {
  pdfUrl: string;
  numPages: number;
  bboxMode: boolean;
  bboxSelections: BBoxSelection[];
  highlightTarget?: {
    page: number;
    bbox_norm: [number, number, number, number];
  } | null;
  onNumPagesLoad: (n: number) => void;
  onBBoxAdd: (sel: BBoxSelection) => void;
  onBackgroundClick?: () => void;
}

export function PdfViewer({
  pdfUrl,
  bboxMode,
  bboxSelections,
  highlightTarget,
  onNumPagesLoad,
  onBBoxAdd,
  onBackgroundClick,
}: PdfViewerProps) {
  const [scrollContainer, setScrollContainer] = useState<HTMLDivElement | null>(
    null
  );
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  const { doc, numPages, pageDims, loading, error } = usePdfDocument(pdfUrl);
  const { visiblePages, observeElement } = useVirtualPages(scrollContainer);
  const { getRenderedCanvas } = useRenderQueue(
    doc,
    pageDims,
    visiblePages,
    PAGE_WIDTH
  );
  const { getTextLayer, getAnnotLayer } = usePageLayers(
    doc,
    pageDims,
    visiblePages,
    PAGE_WIDTH,
    pageRefs.current,
  );
  const { dragState, liveRect, handleDown, handleMove, handleUp } =
    useBBoxDrag(pageDims, onBBoxAdd);

  // Notify parent of page count
  useEffect(() => {
    if (numPages > 0) {
      onNumPagesLoad(numPages);
    }
  }, [numPages, onNumPagesLoad]);

  // Scroll to highlight target
  useEffect(() => {
    if (highlightTarget != null) {
      const el = pageRefs.current.get(highlightTarget.page);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [highlightTarget]);

  // Track page slot refs for scrollIntoView
  const trackPageRef = useCallback(
    (index: number, el: HTMLDivElement | null) => {
      if (el) {
        pageRefs.current.set(index, el);
      } else {
        pageRefs.current.delete(index);
      }
      observeElement(index, el);
    },
    [observeElement]
  );

  if (loading && pageDims.length === 0) {
    return <div className="status">Loading PDF...</div>;
  }

  if (error) {
    return <div className="status">Failed to load PDF from backend.</div>;
  }

  return (
    <div
      className="pdf-scroll-container"
      ref={setScrollContainer}
    >
      {pageDims.map((dim, i) => {
        const pageHeight = PAGE_WIDTH * (dim.h / dim.w);
        return (
          <PageSlot
            key={i}
            pageIndex={i}
            pageWidth={PAGE_WIDTH}
            pageHeight={pageHeight}
            nativeWidth={dim.w}
            offscreenCanvas={getRenderedCanvas(i)}
            textLayerDiv={getTextLayer(i)}
            annotLayerDiv={getAnnotLayer(i)}
            observeElement={trackPageRef}
          >
            <BBoxOverlay
              pageIndex={i}
              bboxMode={bboxMode}
              bboxSelections={bboxSelections}
              highlightTarget={highlightTarget}
              liveRect={liveRect}
              dragPage={dragState?.page ?? null}
              onMouseDown={handleDown}
              onMouseMove={handleMove}
              onMouseUp={handleUp}
              onBackgroundClick={onBackgroundClick}
            />
          </PageSlot>
        );
      })}
    </div>
  );
}
