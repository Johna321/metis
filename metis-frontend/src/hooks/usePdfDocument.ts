import { useState, useRef, useEffect } from "react";
import * as pdfjs from "pdfjs-dist";
import type { PDFDocumentProxy } from "pdfjs-dist";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

export type PageDim = { w: number; h: number };

export function usePdfDocument(pdfUrl: string | null) {
  const docRef = useRef<PDFDocumentProxy | null>(null);
  const [pageDims, setPageDims] = useState<PageDim[]>([]);
  const [numPages, setNumPages] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!pdfUrl) {
      setPageDims([]);
      setNumPages(0);
      return;
    }

    let cancelled = false;
    const prevDoc = docRef.current;

    async function load() {
      setLoading(true);
      setError(null);
      setPageDims([]);
      setNumPages(0);

      // Destroy previous document
      if (prevDoc) {
        prevDoc.destroy();
        docRef.current = null;
      }

      try {
        const loadingTask = pdfjs.getDocument(pdfUrl!);
        const doc = await loadingTask.promise;
        if (cancelled) {
          doc.destroy();
          return;
        }
        docRef.current = doc;

        const n = doc.numPages;
        const dims: PageDim[] = [];
        for (let i = 1; i <= n; i++) {
          const page = await doc.getPage(i);
          const vp = page.getViewport({ scale: 1 });
          dims.push({ w: vp.width, h: vp.height });
        }

        if (cancelled) return;
        setPageDims(dims);
        setNumPages(n);
        setLoading(false);
      } catch (err) {
        if (cancelled) return;
        setError(String(err));
        setLoading(false);
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, [pdfUrl]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      docRef.current?.destroy();
    };
  }, []);

  return { doc: docRef, numPages, pageDims, loading, error };
}
