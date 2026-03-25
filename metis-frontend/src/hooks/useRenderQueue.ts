import { useRef, useEffect, useState, useCallback } from "react";
import type { PDFDocumentProxy, RenderTask } from "pdfjs-dist";

const MAX_CONCURRENT = 3;
const EVICT_DISTANCE = 5;

export function useRenderQueue(
  doc: React.RefObject<PDFDocumentProxy | null>,
  pageDims: Array<{ w: number; h: number }>,
  visiblePages: Set<number>,
  pageWidth: number
) {
  const canvasCache = useRef<Map<number, OffscreenCanvas>>(new Map());
  const inFlight = useRef<Map<number, RenderTask>>(new Map());
  const [renderGeneration, setRenderGeneration] = useState(0);

  useEffect(() => {
    if (!doc.current || pageDims.length === 0) return;

    const currentDoc = doc.current;
    const dpr = window.devicePixelRatio || 1;

    // Cancel renders for pages no longer visible
    for (const [pageIdx, task] of inFlight.current) {
      if (!visiblePages.has(pageIdx)) {
        task.cancel();
        inFlight.current.delete(pageIdx);
      }
    }

    // Evict cached canvases far from any visible page
    if (visiblePages.size > 0) {
      for (const [pageIdx] of canvasCache.current) {
        let minDist = Infinity;
        for (const vp of visiblePages) {
          minDist = Math.min(minDist, Math.abs(pageIdx - vp));
        }
        if (minDist > EVICT_DISTANCE) {
          canvasCache.current.delete(pageIdx);
        }
      }
    }

    // Sort visible pages by distance from center of visible range
    const visibleArr = [...visiblePages];
    const center =
      visibleArr.length > 0
        ? visibleArr.reduce((a, b) => a + b, 0) / visibleArr.length
        : 0;
    const sorted = visibleArr.sort(
      (a, b) => Math.abs(a - center) - Math.abs(b - center)
    );

    // Render pages that need it
    let activeCount = inFlight.current.size;
    for (const pageIdx of sorted) {
      if (activeCount >= MAX_CONCURRENT) break;
      if (canvasCache.current.has(pageIdx) || inFlight.current.has(pageIdx))
        continue;

      const dim = pageDims[pageIdx];
      if (!dim) continue;

      const scale = (pageWidth / dim.w) * dpr;

      activeCount++;

      // Use IIFE to capture pageIdx properly
      (async (idx: number) => {
        try {
          const page = await currentDoc.getPage(idx + 1); // 1-indexed
          // Check if still relevant
          if (!doc.current || doc.current !== currentDoc) return;

          const viewport = page.getViewport({ scale });
          const canvas = new OffscreenCanvas(
            Math.floor(viewport.width),
            Math.floor(viewport.height)
          );
          const ctx = canvas.getContext("2d")!;
          const renderTask = page.render({
            canvas: null as unknown as HTMLCanvasElement,
            canvasContext: ctx as unknown as CanvasRenderingContext2D,
            viewport,
          });

          inFlight.current.set(idx, renderTask);

          await renderTask.promise;
          inFlight.current.delete(idx);
          canvasCache.current.set(idx, canvas);
          setRenderGeneration((g) => g + 1);
        } catch (err: unknown) {
          inFlight.current.delete(idx);
          // RenderTask.cancel() throws — that's expected
          if (
            err &&
            typeof err === "object" &&
            "name" in err &&
            (err as { name: string }).name === "RenderingCancelledException"
          ) {
            return;
          }
          console.warn(`[PDF] render failed for page ${idx}:`, err);
        }
      })(pageIdx);
    }
  }, [doc, pageDims, visiblePages, pageWidth]);

  const getRenderedCanvas = useCallback(
    (pageIdx: number): OffscreenCanvas | undefined => {
      return canvasCache.current.get(pageIdx);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [renderGeneration]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      for (const [, task] of inFlight.current) {
        task.cancel();
      }
      inFlight.current.clear();
      canvasCache.current.clear();
    };
  }, []);

  return { getRenderedCanvas, renderGeneration };
}
