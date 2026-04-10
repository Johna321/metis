import type { BBoxSelection } from "./PdfViewer";
import type { SearchMatch } from "../hooks/usePdfSearch";

type LiveRect = { left: number; top: number; width: number; height: number };

interface BBoxOverlayProps {
  pageIndex: number;
  bboxMode: boolean;
  bboxSelections: BBoxSelection[];
  highlightTarget?: {
    page: number;
    bbox_norm: [number, number, number, number];
  } | null;
  liveRect: LiveRect | null;
  dragPage: number | null; // 0-indexed page of the active drag
  onMouseDown: (e: React.MouseEvent<HTMLDivElement>, pageIndex: number) => void;
  onMouseMove: (e: React.MouseEvent<HTMLDivElement>) => void;
  onMouseUp: (e: React.MouseEvent<HTMLDivElement>, pageIndex: number) => void;
  onBackgroundClick?: () => void;
  searchMatches?: SearchMatch[];
  currentSearchMatchIdx?: number;
}

export function BBoxOverlay({
  pageIndex,
  bboxMode,
  bboxSelections,
  highlightTarget,
  liveRect,
  dragPage,
  onMouseDown,
  onMouseMove,
  onMouseUp,
  onBackgroundClick,
  searchMatches,
  currentSearchMatchIdx,
}: BBoxOverlayProps) {
  const showOverlay = bboxMode || highlightTarget || (searchMatches && searchMatches.length > 0);
  if (!showOverlay) return null;

  return (
    <div
      className="bbox-overlay"
      onMouseDown={(e) => bboxMode && onMouseDown(e, pageIndex)}
      onMouseMove={(e) => bboxMode && onMouseMove(e)}
      onMouseUp={(e) => bboxMode && onMouseUp(e, pageIndex)}
      onClick={() => !bboxMode && onBackgroundClick?.()}
      style={{
        zIndex: bboxMode ? 10 : 3,
        cursor: bboxMode ? "crosshair" : "default",
      }}
    >
      {bboxMode && (
        <>
          {bboxSelections
            .filter((b) => b.page === pageIndex)
            .map((b, idx) => (
              <div
                key={idx}
                className="bbox-completed"
                style={{
                  left: `${b.bbox_norm[0] * 100}%`,
                  top: `${b.bbox_norm[1] * 100}%`,
                  width: `${(b.bbox_norm[2] - b.bbox_norm[0]) * 100}%`,
                  height: `${(b.bbox_norm[3] - b.bbox_norm[1]) * 100}%`,
                }}
              />
            ))}
          {liveRect && dragPage === pageIndex && (
            <div className="bbox-rubber-band" style={liveRect} />
          )}
        </>
      )}
      {highlightTarget?.page === pageIndex && (
        <div
          className="bbox-citation-highlight"
          style={{
            left: `${highlightTarget.bbox_norm[0] * 100}%`,
            top: `${highlightTarget.bbox_norm[1] * 100}%`,
            width: `${(highlightTarget.bbox_norm[2] - highlightTarget.bbox_norm[0]) * 100}%`,
            height: `${(highlightTarget.bbox_norm[3] - highlightTarget.bbox_norm[1]) * 100}%`,
          }}
        />
      )}
      {searchMatches &&
        searchMatches
          .filter((m) => m.page === pageIndex)
          .map((m, idx) => {
            const matchIdx = searchMatches.indexOf(m);
            const isCurrentMatch = matchIdx === currentSearchMatchIdx;
            return (
              <div
                key={idx}
                className={`bbox-search-match ${
                  isCurrentMatch ? "bbox-search-match-active" : ""
                }`}
                style={{
                  left: `${m.bbox_norm[0] * 100}%`,
                  top: `${m.bbox_norm[1] * 100}%`,
                  width: `${(m.bbox_norm[2] - m.bbox_norm[0]) * 100}%`,
                  height: `${(m.bbox_norm[3] - m.bbox_norm[1]) * 100}%`,
                }}
              />
            );
          })}
    </div>
  );
}
