import { useRef, useEffect, useState, useCallback, useMemo } from "react";
import { TextLayer, AnnotationLayer } from "pdfjs-dist";
import type { PDFDocumentProxy } from "pdfjs-dist";

const EVICT_DISTANCE = 5;

/** Minimal IPDFLinkService that opens external links in new tabs and navigates internal links. */
function createLinkService(
  docRef: React.RefObject<PDFDocumentProxy | null>,
  pageRefs: Map<number, HTMLDivElement>,
  numPagesRef: { current: number },
) {
  function scrollToPage(pageIndex: number) {
    const el = pageRefs.get(pageIndex);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  return {
    get pagesCount() { return numPagesRef.current; },
    get page() { return 1; },
    set page(_v: number) {},
    get rotation() { return 0; },
    set rotation(_v: number) {},
    get isInPresentationMode() { return false; },
    get externalLinkEnabled() { return true; },
    set externalLinkEnabled(_v: boolean) {},
    async goToDestination(dest: unknown) {
      const currentDoc = docRef.current;
      if (!currentDoc) return;
      try {
        // Named destination (string) → resolve to explicit dest array
        let explicitDest = dest;
        if (typeof dest === "string") {
          explicitDest = await currentDoc.getDestination(dest);
        }
        if (!Array.isArray(explicitDest) || explicitDest.length === 0) return;
        const pageRef = explicitDest[0];
        const pageIndex = await currentDoc.getPageIndex(pageRef);
        scrollToPage(pageIndex);
      } catch (err) {
        console.warn("[PDF] goToDestination failed:", err);
      }
    },
    goToPage(val: number | string) {
      const pageNum = typeof val === "string" ? parseInt(val, 10) : val;
      if (Number.isFinite(pageNum) && pageNum >= 1) {
        scrollToPage(pageNum - 1); // convert 1-indexed to 0-indexed
      }
    },
    goToXY(_pageNumber: number, _x: number, _y: number) {},
    addLinkAttributes(link: HTMLAnchorElement, url: string, newWindow?: boolean) {
      link.href = url;
      link.rel = "noopener noreferrer nofollow";
      if (newWindow || /^https?:\/\//.test(url)) {
        link.target = "_blank";
      }
    },
    getDestinationHash(_dest: unknown) { return "#"; },
    getAnchorUrl(hash: string) { return hash; },
    setHash(_hash: string) {},
    executeNamedAction(_action: string) {},
    executeSetOCGState(_action: unknown) {},
  };
}

type LayerEntry = {
  textDiv: HTMLDivElement;
  annotDiv: HTMLDivElement;
  textLayer?: TextLayer;
};

export function usePageLayers(
  doc: React.RefObject<PDFDocumentProxy | null>,
  pageDims: Array<{ w: number; h: number }>,
  visiblePages: Set<number>,
  pageWidth: number,
  pageRefs: Map<number, HTMLDivElement>,
) {
  const layerCache = useRef<Map<number, LayerEntry>>(new Map());
  const inFlight = useRef<Set<number>>(new Set());
  const [layerGeneration, setLayerGeneration] = useState(0);
  const numPagesRef = useRef(0);
  numPagesRef.current = pageDims.length;
  const linkService = useMemo(() => createLinkService(doc, pageRefs, numPagesRef), [doc, pageRefs]);

  useEffect(() => {
    if (!doc.current || pageDims.length === 0) return;
    const currentDoc = doc.current;

    // Evict layers far from visible pages
    if (visiblePages.size > 0) {
      for (const [pageIdx, entry] of layerCache.current) {
        let minDist = Infinity;
        for (const vp of visiblePages) {
          minDist = Math.min(minDist, Math.abs(pageIdx - vp));
        }
        if (minDist > EVICT_DISTANCE) {
          entry.textLayer?.cancel();
          entry.textDiv.innerHTML = "";
          entry.annotDiv.innerHTML = "";
          layerCache.current.delete(pageIdx);
        }
      }
    }

    // Build layers for visible pages
    for (const pageIdx of visiblePages) {
      if (layerCache.current.has(pageIdx) || inFlight.current.has(pageIdx)) continue;
      const dim = pageDims[pageIdx];
      if (!dim) continue;

      inFlight.current.add(pageIdx);

      (async (idx: number) => {
        try {
          const page = await currentDoc.getPage(idx + 1);
          if (!doc.current || doc.current !== currentDoc) return;

          const scale = pageWidth / dim.w;
          const viewport = page.getViewport({ scale });

          // -- TextLayer --
          const textDiv = document.createElement("div");
          textDiv.className = "textLayer";
          const textLayer = new TextLayer({
            textContentSource: page.streamTextContent(),
            container: textDiv,
            viewport,
          });
          await textLayer.render();

          // -- AnnotationLayer --
          const annotDiv = document.createElement("div");
          annotDiv.className = "annotationLayer";

          const annotations = await page.getAnnotations();

          const annotLayer = new AnnotationLayer({
            div: annotDiv,
            page,
            viewport,
            linkService: linkService as any,
            accessibilityManager: null,
            annotationCanvasMap: null,
            annotationEditorUIManager: null,
            structTreeLayer: null,
            commentManager: null,
            annotationStorage: null,
          });

          await annotLayer.render({
            viewport,
            div: annotDiv,
            annotations,
            page,
            linkService: linkService as any,
            renderForms: false,
          });

          inFlight.current.delete(idx);
          layerCache.current.set(idx, { textDiv, annotDiv, textLayer });
          setLayerGeneration((g) => g + 1);
        } catch (err) {
          inFlight.current.delete(idx);
          console.warn(`[PDF] layer build failed for page ${idx}:`, err);
        }
      })(pageIdx);
    }
  }, [doc, pageDims, visiblePages, pageWidth, linkService]);

  const getTextLayer = useCallback(
    (pageIdx: number): HTMLDivElement | undefined => {
      return layerCache.current.get(pageIdx)?.textDiv;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [layerGeneration],
  );

  const getAnnotLayer = useCallback(
    (pageIdx: number): HTMLDivElement | undefined => {
      return layerCache.current.get(pageIdx)?.annotDiv;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [layerGeneration],
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      for (const [, entry] of layerCache.current) {
        entry.textLayer?.cancel();
      }
      layerCache.current.clear();
      inFlight.current.clear();
    };
  }, []);

  return { getTextLayer, getAnnotLayer };
}
