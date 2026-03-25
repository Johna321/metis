import { useRef, useEffect, useCallback } from "react";

interface PageSlotProps {
  pageIndex: number;
  pageWidth: number;
  pageHeight: number;
  nativeWidth: number;
  offscreenCanvas: OffscreenCanvas | undefined;
  textLayerDiv?: HTMLDivElement;
  annotLayerDiv?: HTMLDivElement;
  observeElement: (index: number, el: HTMLDivElement | null) => void;
  children?: React.ReactNode;
}

export function PageSlot({
  pageIndex,
  pageWidth,
  pageHeight,
  nativeWidth,
  offscreenCanvas,
  textLayerDiv,
  annotLayerDiv,
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

  // Mount/unmount text layer div
  useEffect(() => {
    const wrap = divRef.current;
    if (!wrap || !textLayerDiv) return;
    wrap.appendChild(textLayerDiv);
    return () => {
      if (textLayerDiv.parentNode === wrap) wrap.removeChild(textLayerDiv);
    };
  }, [textLayerDiv]);

  // Mount/unmount annotation layer div
  useEffect(() => {
    const wrap = divRef.current;
    if (!wrap || !annotLayerDiv) return;
    wrap.appendChild(annotLayerDiv);
    return () => {
      if (annotLayerDiv.parentNode === wrap) wrap.removeChild(annotLayerDiv);
    };
  }, [annotLayerDiv]);

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
      style={{
        width: pageWidth,
        height: pageHeight,
        position: "relative",
        "--scale-factor": pageWidth / nativeWidth,
        "--total-scale-factor": "var(--scale-factor)",
        "--scale-round-x": "1px",
        "--scale-round-y": "1px",
      } as React.CSSProperties}
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
