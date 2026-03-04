import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

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

export type BboxSelection = {
  page: number;
  bbox_norm: [number, number, number, number];
};

export async function ingestPdf(filePath: string): Promise<IngestResponse> {
  return invoke("ingest_pdf", { filePath });
}

export async function retrieveEvidence(
  docId: string,
  pageZeroBased: number,
  selectedText: string
): Promise<EvidenceItem[]> {
  return invoke("retrieve_evidence", {
    docId,
    page: pageZeroBased,
    selectedText,
  });
}

export async function getDocumentPdfUrl(docId: string): Promise<string> {
  return invoke("get_document_pdf_url", { docId });
}

// Chat streaming types and callback

export type ChatStreamEvent =
  | { kind: "TextDelta"; text: string }
  | { kind: "ToolCallStart"; name: string }
  | { kind: "ToolCallDelta"; text: string }
  | { kind: "ToolCallDone"; id: string; name: string; arguments: unknown }
  | { kind: "MessageDone"; role: string; content: string | null; tool_calls: unknown | null }
  | { kind: "Error"; message: string };

export type ChatStreamCallbacks = {
  onTextDelta?: (text: string) => void;
  onToolCallStart?: (name: string) => void;
  onToolCallDelta?: (text: string) => void;
  onToolCallDone?: (id: string, name: string, args: unknown) => void;
  onMessageDone?: (role: string, content: string | null, toolCalls: unknown | null) => void;
  onError?: (message: string) => void;
};

/**
 * Start a chat stream. Returns an unlisten function to stop receiving events.
 *
 * The Rust side spawns an async task that reads SSE from the backend and emits
 * "chat-stream" Tauri events. This function wires those events to callbacks.
 *
 * Example usage in a React component:
 *
 *   const [response, setResponse] = useState("");
 *   const unlistenRef = useRef<UnlistenFn | null>(null);
 *
 *   async function sendMessage(text: string) {
 *     setResponse("");
 *     unlistenRef.current = await chatStart(docId, text, {
 *       onTextDelta: (t) => setResponse((prev) => prev + t),
 *       onToolCallStart: (name) => console.log("calling tool:", name),
 *       onError: (msg) => setStatus(`Error: ${msg}`),
 *       onMessageDone: () => {
 *         // Stream finished — clean up listener
 *         unlistenRef.current?.();
 *         unlistenRef.current = null;
 *       },
 *     });
 *   }
 *
 *   // Clean up on unmount
 *   useEffect(() => () => { unlistenRef.current?.(); }, []);
 */
export async function chatStart(
  docId: string,
  message: string,
  callbacks: ChatStreamCallbacks,
  opts?: { provider?: string; model?: string; selections?: BboxSelection[] },
): Promise<UnlistenFn> {
  const unlisten = await listen<ChatStreamEvent>("chat-stream", (event) => {
    const ev = event.payload;
    switch (ev.kind) {
      case "TextDelta":
        callbacks.onTextDelta?.(ev.text);
        break;
      case "ToolCallStart":
        callbacks.onToolCallStart?.(ev.name);
        break;
      case "ToolCallDelta":
        callbacks.onToolCallDelta?.(ev.text);
        break;
      case "ToolCallDone":
        callbacks.onToolCallDone?.(ev.id, ev.name, ev.arguments);
        break;
      case "MessageDone":
        callbacks.onMessageDone?.(ev.role, ev.content, ev.tool_calls);
        break;
      case "Error":
        callbacks.onError?.(ev.message);
        break;
    }
  });

  await invoke("chat_start", {
    docId,
    message,
    provider: opts?.provider ?? null,
    model: opts?.model ?? null,
    selections: opts?.selections ?? null,
  });

  return unlisten;
}
