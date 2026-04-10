import { useState, useCallback } from "react";
import type { PDFDocumentProxy } from "pdfjs-dist";
import type { PageDim } from "./usePdfDocument";

export type SearchMatch = {
  page: number;
  bbox_norm: [number, number, number, number];
};

export function usePdfSearch(
  doc: React.RefObject<PDFDocumentProxy | null>,
  pageDims: PageDim[]
) {
  const [matches, setMatches] = useState<SearchMatch[]>([]);
  const [currentMatchIdx, setCurrentMatchIdx] = useState(0);
  const [searchQuery, setSearchQuery] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);

  const search = useCallback(
    async (query: string) => {
      if (!doc.current || !query.trim()) return;
      setIsSearching(true);
      setSearchQuery(query);

      const queryLower = query.toLowerCase();
      const allMatches: SearchMatch[] = [];

      for (let pageIdx = 0; pageIdx < pageDims.length; pageIdx++) {
        try {
          const page = await doc.current.getPage(pageIdx + 1);
          const textContent = await page.getTextContent();
          const dim = pageDims[pageIdx];

          // collect text items and track their string positions
          const items = textContent.items.filter(
            (item) => "str" in item
          );

          // build concatenated page text and track item boundaries
          let pageText = "";
          const itemRanges: Array<{
            start: number;
            end: number;
            item: any;
          }> = [];

          for (const item of items) {
            if (!("str" in item)) continue;
            const start = pageText.length;
            pageText += item.str;
            const end = pageText.length;
            itemRanges.push({ start, end, item });
          }

          // search for matches in page text
          const pageTextLower = pageText.toLowerCase();
          let pos = 0;

          while ((pos = pageTextLower.indexOf(queryLower, pos)) !== -1) {
            const matchEnd = pos + queryLower.length;

            // find all items that overlap with the match range
            const overlapping = itemRanges.filter(
              (r) => r.end > pos && r.start < matchEnd
            );

            if (overlapping.length > 0) {
              // compute merged bbox from overlapping items
              let x0 = Infinity;
              let y0_pdf = Infinity;
              let x1 = -Infinity;
              let y1_pdf = -Infinity;

              for (const { item } of overlapping) {
                const tx = item.transform[4];
                const ty = item.transform[5];
                const iw = item.width;
                const ih = item.height;

                x0 = Math.min(x0, tx);
                y0_pdf = Math.min(y0_pdf, ty);
                x1 = Math.max(x1, tx + iw);
                y1_pdf = Math.max(y1_pdf, ty + ih);
              }

              // convert from PDF space (y-up, origin bottom-left) to normalized space (y-down, origin top-left)
              allMatches.push({
                page: pageIdx,
                bbox_norm: [
                  x0 / dim.w,
                  1 - y1_pdf / dim.h,
                  x1 / dim.w,
                  1 - y0_pdf / dim.h,
                ] as [number, number, number, number],
              });
            }

            pos++;
          }
        } catch (err) {
          console.error(
            `Error searching page ${pageIdx}:`,
            err
          );
        }
      }

      setMatches(allMatches);
      setCurrentMatchIdx(0);
      setIsSearching(false);
    },
    [doc, pageDims]
  );

  const clearSearch = useCallback(() => {
    setMatches([]);
    setCurrentMatchIdx(0);
    setSearchQuery(null);
  }, []);

  const nextMatch = useCallback(() => {
    if (matches.length === 0) return;
    setCurrentMatchIdx((i) => (i + 1) % matches.length);
  }, [matches.length]);

  const prevMatch = useCallback(() => {
    if (matches.length === 0) return;
    setCurrentMatchIdx((i) => (i - 1 + matches.length) % matches.length);
  }, [matches.length]);

  return {
    search,
    clearSearch,
    matches,
    currentMatchIdx,
    searchQuery,
    isSearching,
    nextMatch,
    prevMatch,
  };
}
