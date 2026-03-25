import { useRef, useEffect, useCallback } from "react";

interface PageSlotProps {
  pageIndex: number;
  pageWidth: number;
  pageHeight: number;
  offscreenCanvas: OffscreenCanvas | undefined;
  observeElement: (index: number, el: HTMLDivElement | null) => void;
  children?: React.ReactNode;
}

export function PageSlot({
  pageIndex,
  pageWidth,
  pageHeight,
  offscreenCanvas,
  observeElement,
  children,
}: PageSlotProps) {
  const divRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const setRef = useCallback(
    (el: HTMLDivElement | null) => {
      divRef.current = el;
      observeElement(pageIndex, el);
    },
    [pageIndex, observeElement]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !offscreenCanvas) return;

    canvas.width = offscreenCanvas.width;
    canvas.height = offscreenCanvas.height;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(offscreenCanvas, 0, 0);
    }
  }, [offscreenCanvas]);

  return (
    <div
      className="page-wrap"
      ref={setRef}
      style={{ width: pageWidth, height: pageHeight, position: "relative" }}
    >
      {offscreenCanvas ? (
        <canvas
          ref={canvasRef}
          style={{ width: pageWidth, height: pageHeight, display: "block" }}
        />
      ) : (
        <div
          className="page-placeholder"
          style={{ width: pageWidth, height: pageHeight }}
        >
          {pageIndex + 1}
        </div>
      )}
      {children}
    </div>
  );
}
