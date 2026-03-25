import { useRef, useCallback, useState, useEffect } from "react";

export function useVirtualPages(scrollContainer: HTMLDivElement | null) {
  const [visiblePages, setVisiblePages] = useState<Set<number>>(new Set());
  const observerRef = useRef<IntersectionObserver | null>(null);
  const elementMap = useRef<Map<number, Element>>(new Map());

  useEffect(() => {
    if (!scrollContainer) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        setVisiblePages((prev) => {
          const next = new Set(prev);
          for (const entry of entries) {
            const idx = Number(entry.target.getAttribute("data-page-index"));
            if (entry.isIntersecting) {
              next.add(idx);
            } else {
              next.delete(idx);
            }
          }
          // Only create a new set if something changed
          if (next.size === prev.size && [...next].every((v) => prev.has(v))) {
            return prev;
          }
          return next;
        });
      },
      {
        root: scrollContainer,
        rootMargin: "1200px 0px",
      }
    );

    // Re-observe any elements already registered
    for (const [, el] of elementMap.current) {
      observerRef.current.observe(el);
    }

    return () => {
      observerRef.current?.disconnect();
      observerRef.current = null;
    };
  }, [scrollContainer]);

  const observeElement = useCallback(
    (index: number, el: HTMLDivElement | null) => {
      const prev = elementMap.current.get(index);
      if (prev && observerRef.current) {
        observerRef.current.unobserve(prev);
      }

      if (el) {
        elementMap.current.set(index, el);
        el.setAttribute("data-page-index", String(index));
        if (observerRef.current) {
          observerRef.current.observe(el);
        }
      } else {
        elementMap.current.delete(index);
      }
    },
    []
  );

  return { visiblePages, observeElement };
}
