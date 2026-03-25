import { useState, useCallback } from "react";
import type { BBoxSelection } from "../components/PdfViewer";

const MIN_DRAG_PX = 5;

type DragState = {
  page: number; // 0-indexed page
  startX: number;
  startY: number;
};

type LiveRect = { left: number; top: number; width: number; height: number };

export function useBBoxDrag(
  pageDims: Array<{ w: number; h: number }>,
  onBBoxAdd: (sel: BBoxSelection) => void
) {
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [liveRect, setLiveRect] = useState<LiveRect | null>(null);

  const handleDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>, pageIndex: number) => {
      e.preventDefault();
      const rect = e.currentTarget.getBoundingClientRect();
      setDragState({
        page: pageIndex,
        startX: e.clientX - rect.left,
        startY: e.clientY - rect.top,
      });
      setLiveRect(null);
    },
    []
  );

  const handleMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!dragState) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const curX = e.clientX - rect.left;
      const curY = e.clientY - rect.top;
      setLiveRect({
        left: Math.min(dragState.startX, curX),
        top: Math.min(dragState.startY, curY),
        width: Math.abs(curX - dragState.startX),
        height: Math.abs(curY - dragState.startY),
      });
    },
    [dragState]
  );

  const handleUp = useCallback(
    (e: React.MouseEvent<HTMLDivElement>, pageIndex: number) => {
      if (!dragState || dragState.page !== pageIndex) return;

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

      const dims = pageDims[pageIndex];
      const bbox_pdf: [number, number, number, number] = dims
        ? [x0n * dims.w, y0n * dims.h, x1n * dims.w, y1n * dims.h]
        : [0, 0, 0, 0];

      onBBoxAdd({
        page: pageIndex,
        bbox_norm: [x0n, y0n, x1n, y1n],
        bbox_pdf,
      });
    },
    [dragState, pageDims, onBBoxAdd]
  );

  return { dragState, liveRect, handleDown, handleMove, handleUp };
}
