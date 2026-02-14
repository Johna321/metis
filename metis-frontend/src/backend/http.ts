export type IngestResponse = {
  doc_id: string;
  n_pages: number;
  n_spans: number;
  ingest: Record<string, unknown>;
};

export type EvidenceItem = {
  span_id: string;
  page: number;
  bbox_norm: [number, number, number, number]; // (x0,y0,x1,y1) normalized 0..1
  text: string;
  score: number;
};

export async function ingestPdf(baseUrl: string, file: File): Promise<IngestResponse> {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch(`${baseUrl}/ingest?engine=blocks`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function retrieveEvidence(
  baseUrl: string,
  docId: string,
  pageZeroBased: number,
  selectedText: string
): Promise<EvidenceItem[]> {
  const res = await fetch(`${baseUrl}/retrieve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: docId, page: pageZeroBased, selected_text: selectedText }),
  });

  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function documentPdfUrl(baseUrl: string, docId: string): string {
  return `${baseUrl}/documents/${docId}/pdf`;
}
