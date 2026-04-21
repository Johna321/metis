import { useRef, useEffect, useState, useCallback } from "react";

import "pdfjs-dist/web/pdf_viewer.css";

import { usePdfDocument } from "../hooks/usePdfDocument";
import { useVirtualPages } from "../hooks/useVirtualPages";
import { useRenderQueue } from "../hooks/useRenderQueue";
import { usePageLayers } from "../hooks/usePageLayers";
import { useBBoxDrag } from "../hooks/useBBoxDrag";
import { usePdfSearch } from "../hooks/usePdfSearch";
import { PageSlot } from "./PageSlot";
import { BBoxOverlay } from "./BBoxOverlay";
import { NoteMarkers, type NoteMarkersProps } from "./NoteMarkers";
import type { Note } from "./NotesPanel";

const PAGE_WIDTH = 900;

export type BBoxSelection = {
  page: number; // 0-indexed
  bbox_norm: [number, number, number, number]; // [x0, y0, x1, y1] in [0, 1]
  bbox_pdf: [number, number, number, number]; // [x0, y0, x1, y1] in PDF points (PyMuPDF)
};

export type PageDims = { w: number; h: number };

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
  onContextTextChange?: (text: string | null) => void;
  notes?: Note[];
  expandedNoteId?: string | null;
  onSelectNote?: (noteId: string) => void;
}

export function PdfViewer({
  pdfUrl,
  bboxMode,
  bboxSelections,
  highlightTarget,
  onNumPagesLoad,
  onBBoxAdd,
  onBackgroundClick,
  onContextTextChange,
  notes = [],
  expandedNoteId = null,
  onSelectNote,
}: PdfViewerProps) {
  const [scrollContainer, setScrollContainer] = useState<HTMLDivElement | null>(
    null
  );
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; text: string } | null>(null);
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const scrollContainerRef = useRef<HTMLDivElement>(null);

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
  const { search, clearSearch, matches, currentMatchIdx, searchQuery, nextMatch, prevMatch } = usePdfSearch(doc, pageDims);

  // calculate page positions (X,Y offset of each page in scroll container)
  const pagePositions = pageDims.map((_, idx) => {
    const pageEl = pageRefs.current.get(idx);
    let x = 0;
    let y = 0;
    if (pageEl && scrollContainer) {
      const pageRect = pageEl.getBoundingClientRect();
      const scrollRect = scrollContainer.getBoundingClientRect();
      x = pageRect.left - scrollRect.left + scrollContainer.scrollLeft;
      y = pageRect.top - scrollRect.top + scrollContainer.scrollTop;
    }
    return { x, y };
  });

  const { dragState, liveRect, handleDown, handleMove, handleUp } =
    useBBoxDrag(pageDims, scrollContainer, pagePositions, onBBoxAdd);

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

  // handle right-click on text selection
  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      const selectedText = window.getSelection()?.toString();
      if (!selectedText || !onContextTextChange) return;

      e.preventDefault();
      setContextMenu({
        x: e.clientX,
        y: e.clientY,
        text: selectedText,
      });
    },
    [onContextTextChange]
  );

  // handle "Add to context" menu item click
  const handleAddToContext = useCallback(() => {
    if (!contextMenu || !onContextTextChange) return;

    const maxLength = 5000;
    const truncatedText =
      contextMenu.text.length > maxLength ? contextMenu.text.slice(0, maxLength) : contextMenu.text;
    onContextTextChange(truncatedText);
    setContextMenu(null);
  }, [contextMenu, onContextTextChange]);

  // handle "Copy" menu item click
  const handleCopyText = useCallback(() => {
    if (!contextMenu) return;

    navigator.clipboard.writeText(contextMenu.text);
    setContextMenu(null);
  }, [contextMenu]);

  // handle "Find all" menu item click
  const handleFindAll = useCallback(async () => {
    if (!contextMenu) return;
    await search(contextMenu.text);
    setContextMenu(null);
  }, [contextMenu, search]);

  // close menu when clicking elsewhere
  useEffect(() => {
    if (!contextMenu) return;

    const handleClick = () => setContextMenu(null);
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, [contextMenu]);

  // auto-scroll to current search match
  useEffect(() => {
    if (matches.length === 0) return;
    const match = matches[currentMatchIdx];
    const el = pageRefs.current.get(match.page);
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [currentMatchIdx, matches]);

  if (loading && pageDims.length === 0) {
    return <div className="status">Loading PDF...</div>;
  }

  if (error) {
    return <div className="status">Failed to load PDF from backend.</div>;
  }

  return (
    <div className="pdf-viewer-wrapper">
      <div
        className="pdf-scroll-container"
        ref={(el) => {
          setScrollContainer(el);
          scrollContainerRef.current = el;
        }}
        onContextMenu={handleContextMenu}
      >
        {contextMenu && (
          <div
            className="context-menu"
            style={{
              position: "fixed",
              top: `${contextMenu.y}px`,
              left: `${contextMenu.x}px`,
              zIndex: 1000,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button className="context-menu-item" onClick={handleAddToContext}>
              Add to context
            </button>
            <button className="context-menu-item" onClick={handleCopyText}>
              Copy
            </button>
            <button className="context-menu-item" onClick={handleFindAll}>
              Find all
            </button>
          </div>
        )}
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
                searchMatches={matches}
                currentSearchMatchIdx={currentMatchIdx}
              />
            </PageSlot>
          );
        })}
      </div>
      {searchQuery && (
        <div className="search-nav-bar">
          <button
            className="search-close-btn"
            onClick={clearSearch}
            title="Close search"
          >
            x
          </button>
          <span className="search-query" title={searchQuery}>
            "{searchQuery}"
          </span>
          <div className="search-controls">
            <button
              className="search-nav-btn"
              onClick={prevMatch}
              disabled={matches.length === 0}
              title="Previous match"
            >
              ▲
            </button>
            <span className="search-counter">
              {matches.length === 0
                ? "no matches"
                : `${currentMatchIdx + 1} of ${matches.length}`}
            </span>
            <button
              className="search-nav-btn"
              onClick={nextMatch}
              disabled={matches.length === 0}
              title="Next match"
            >
              ▼
            </button>
          </div>
        </div>
      )}
      {notes.length > 0 && (
        <NoteMarkers
          notes={notes}
          expandedNoteId={expandedNoteId}
          onSelectNote={onSelectNote || (() => {})}
          scrollContainerRef={scrollContainerRef}
          pagePositions={pagePositions}
          pageDims={pageDims}
        />
      )}
    </div>
  );
}
