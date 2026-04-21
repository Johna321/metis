import { useMemo, useRef, useEffect, useState } from "react";
import type { Note } from "./NotesPanel";
import type { PageDims } from "./PdfViewer";

interface NoteMarkersProps {
  notes: Note[];
  expandedNoteId: string | null;
  onSelectNote: (noteId: string) => void;
  scrollContainerRef: React.RefObject<HTMLDivElement>;
  pagePositions: Array<{ x: number; y: number }>;
  pageDims: PageDims[];
}

const PDF_PAGE_WIDTH = 900;  // matches PAGE_WIDTH in PdfViewer

export function NoteMarkers({
  notes,
  expandedNoteId,
  onSelectNote,
  scrollContainerRef,
  pagePositions,
  pageDims,
}: NoteMarkersProps) {
  const [scrollTop, setScrollTop] = useState(0);
  const [containerRect, setContainerRect] = useState<DOMRect | null>(null);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const wrapper = container.parentElement;
    if (!wrapper) return;

    function handleScroll() {
      setScrollTop(container.scrollTop);
    }

    function updateRect() {
      const wrapperRect = wrapper.getBoundingClientRect();
      setContainerRect(wrapperRect);
    }

    // update rect on scroll and resize
    container.addEventListener("scroll", handleScroll);
    window.addEventListener("resize", updateRect);

    // use ResizeObserver to detect when panels open/close
    const observer = new ResizeObserver(updateRect);
    observer.observe(wrapper);

    updateRect();

    return () => {
      container.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", updateRect);
      observer.disconnect();
    };
  }, [scrollContainerRef]);

  // force visibility recalculation when notes change
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    // re-read scroll position to ensure markers visibility updates immediately
    setScrollTop(container.scrollTop);
  }, [notes, scrollContainerRef]);

  const markers = useMemo(() => {
    const VISIBILITY_BUFFER = 200; // pixels outside viewport to render

    return notes.map((note) => {
      const pageIdx = note.bbox.page;
      const pagePos = pagePositions[pageIdx];
      const pageDim = pageDims[pageIdx];

      if (!pagePos || !pageDim) {
        return { id: note.id, y: 0, isExpanded: note.id === expandedNoteId, isVisible: false };
      }

      // calculate page height based on aspect ratio
      const pageHeight = PDF_PAGE_WIDTH * (pageDim.h / pageDim.w);

      // center marker within the bbox vertically
      const yTop = note.bbox.bbox_norm[1];
      const yBottom = note.bbox.bbox_norm[3];
      const normalizedY = (yTop + yBottom) / 2;

      // absolute position: page's Y offset + normalized position within page
      const absoluteY = pagePos.y + normalizedY * pageHeight;
      const visibleY = absoluteY - scrollTop;

      // check if marker is within visible range
      const isVisible = visibleY > -VISIBILITY_BUFFER && visibleY < window.innerHeight + VISIBILITY_BUFFER;

      return {
        id: note.id,
        y: visibleY,
        isExpanded: note.id === expandedNoteId,
        isVisible,
      };
    });
  }, [notes, scrollTop, expandedNoteId, pagePositions, pageDims]);

  // calculate left position based on scroll container's left edge
  const leftPos = 0;//containerRect ? Math.max(0, containerRect.left - 32) : -32;

  return (
    <div className="note-markers-container" style={{ left: `${leftPos}px` }}>
      {markers.map((marker) => marker.isVisible && (
        <button
          key={marker.id}
          className={`note-marker ${marker.isExpanded ? "expanded" : ""}`}
          style={{ top: `${marker.y}px` }}
          onClick={() => onSelectNote(marker.id)}
          aria-label="Open note"
          title="Click to view note"
        />
      ))}
    </div>
  );
}
