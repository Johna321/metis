import { useState, useCallback } from "react";
import type { BBoxSelection } from "../components/PdfViewer";

const MIN_DRAG_PX = 5;
const PAGE_WIDTH = 900;

type DragState = {
  page: number; // 0-indexed page where drag started
  startX: number; // absolute X in scroll container coordinates
  startY: number; // absolute Y in scroll container coordinates
};

type LiveRect = { left: number; top: number; width: number; height: number };
type PagePos = { x: number; y: number };

export function useBBoxDrag(
  pageDims: Array<{ w: number; h: number }>,
  scrollContainer: HTMLDivElement | null,
  pagePositions: PagePos[], // {x, y} offset of each page in scroll container
  onBBoxAdd: (sel: BBoxSelection) => void
) {
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [liveRect, setLiveRect] = useState<LiveRect | null>(null);

  const handleDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>, pageIndex: number) => {
      if (!scrollContainer) return;
      e.preventDefault();

      const scrollRect = scrollContainer.getBoundingClientRect();
      const absX = e.clientX - scrollRect.left + scrollContainer.scrollLeft;
      const absY = e.clientY - scrollRect.top + scrollContainer.scrollTop;

      setDragState({
        page: pageIndex,
        startX: absX,
        startY: absY,
      });
      setLiveRect(null);
    },
    [scrollContainer]
  );

  const handleMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!dragState || !scrollContainer) return;

      const scrollRect = scrollContainer.getBoundingClientRect();
      const absX = e.clientX - scrollRect.left + scrollContainer.scrollLeft;
      const absY = e.clientY - scrollRect.top + scrollContainer.scrollTop;

      // calculate absolute bounding box
      const absLeft = Math.min(dragState.startX, absX);
      const absTop = Math.min(dragState.startY, absY);
      const absRight = Math.max(dragState.startX, absX);
      const absBottom = Math.max(dragState.startY, absY);

      // convert to starting page's coordinate system for live preview
      const startPagePos = pagePositions[dragState.page];
      const relLeft = absLeft - startPagePos.x;
      const relTop = absTop - startPagePos.y;
      const relWidth = absRight - absLeft;
      const relHeight = absBottom - absTop;

      setLiveRect({
        left: relLeft,
        top: relTop,
        width: relWidth,
        height: relHeight,
      });
    },
    [dragState, scrollContainer, pagePositions]
  );

  const handleUp = useCallback(
    (e: React.MouseEvent<HTMLDivElement>, _pageIndex: number) => {
      if (!dragState || !scrollContainer) return;

      const scrollRect = scrollContainer.getBoundingClientRect();
      const absX = e.clientX - scrollRect.left + scrollContainer.scrollLeft;
      const absY = e.clientY - scrollRect.top + scrollContainer.scrollTop;

      const dw = Math.abs(absX - dragState.startX);
      const dh = Math.abs(absY - dragState.startY);

      setDragState(null);
      setLiveRect(null);

      if (dw < MIN_DRAG_PX || dh < MIN_DRAG_PX) return;

      // absolute bounding box
      const absLeft = Math.min(dragState.startX, absX);
      const absTop = Math.min(dragState.startY, absY);
      const absRight = Math.max(dragState.startX, absX);
      const absBottom = Math.max(dragState.startY, absY);

      // find the first page that has content within the drag box
      let startPageIdx = 0;
      for (let i = 0; i < pagePositions.length; i++) {
        const pageTop = pagePositions[i].y;
        const pageHeight =
          i < pagePositions.length - 1
            ? pagePositions[i + 1].y - pageTop
            : PAGE_WIDTH * (pageDims[i]?.h ?? 0) / (pageDims[i]?.w ?? 1);
        if (pageTop + pageHeight > absTop) {
          startPageIdx = i;
          break;
        }
      }

      // find the last page whose top edge is before the drag box bottom
      let endPageIdx = pagePositions.length - 1;
      for (let i = 0; i < pagePositions.length; i++) {
        const pageTop = pagePositions[i].y;
        if (pageTop >= absBottom) {
          endPageIdx = i - 1;
          break;
        }
      }

      // create a selection for each affected page
      for (let pageIdx = startPageIdx; pageIdx <= endPageIdx && pageIdx < pageDims.length; pageIdx++) {
        const dims = pageDims[pageIdx];
        if (!dims) continue;

        const pagePos = pagePositions[pageIdx];
        const pagePosTop = pagePos.y;
        const pageHeight =
          pageIdx < pagePositions.length - 1
            ? pagePositions[pageIdx + 1].y - pagePosTop
            : PAGE_WIDTH * (dims.h / dims.w);

        // calculate intersection of drag box with this page
        const intersectTop = Math.max(absTop, pagePosTop);
        const intersectBottom = Math.min(absBottom, pagePosTop + pageHeight);

        if (intersectTop >= intersectBottom) continue;  // no intersection

        const renderedW = PAGE_WIDTH;
        const renderedH = renderedW * (dims.h / dims.w);

        const pagePosLeft = pagePos.x;
        const pageRelLeft = absLeft - pagePosLeft;
        const pageRelTop = intersectTop - pagePosTop;
        const pageRelRight = absRight - pagePosLeft;
        const pageRelBottom = intersectBottom - pagePosTop;

        let x0n = pageRelLeft / renderedW;
        let y0n = pageRelTop / renderedH;
        let x1n = pageRelRight / renderedW;
        let y1n = pageRelBottom / renderedH;

        // clamp to [0, 1]
        x0n = Math.max(0, Math.min(1, x0n));
        y0n = Math.max(0, Math.min(1, y0n));
        x1n = Math.max(0, Math.min(1, x1n));
        y1n = Math.max(0, Math.min(1, y1n));

        const bbox_pdf: [number, number, number, number] = [
          x0n * dims.w,
          y0n * dims.h,
          x1n * dims.w,
          y1n * dims.h,
        ];

        onBBoxAdd({
          page: pageIdx,
          bbox_norm: [x0n, y0n, x1n, y1n],
          bbox_pdf,
        });
      }
    },
    [dragState, scrollContainer, pagePositions, pageDims, onBBoxAdd]
  );

  return { dragState, liveRect, handleDown, handleMove, handleUp };
}
