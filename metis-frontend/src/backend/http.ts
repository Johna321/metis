import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type {
  IngestResponse,
  VectorizeResponse,
  EvidenceItem,
  BboxSelection,
  ChatStreamEvent,
} from "./generated_types";

export interface ApiSettings {
  provider: string;
  model: string;
  anthropic_api_key?: string;
  openai_api_key?: string;
  openrouter_api_key?: string;
  tavily_api_key?: string;
}

import type {
  IngestResponse,
  VectorizeResponse,
  EvidenceItem,
  BboxSelection,
  ChatStreamEvent,
} from "./generated_types";

/** Compute the SHA256 doc_id for a local file (runs in Rust, very fast). */
export async function hashFile(filePath: string): Promise<string> {
  return invoke("hash_file", { filePath });
}

/**
 * Check if a document has already been ingested.
 * Returns the stored metadata if found, or null if not yet ingested.
 */
export async function checkDoc(docId: string): Promise<IngestResponse | null> {
  return invoke("check_doc", { docId });
}

export async function ingestPdf(filePath: string): Promise<IngestResponse> {
  return invoke("ingest_pdf", { filePath });
}

export async function vectorizeDoc(docId: String): Promise<VectorizeResponse> {
  return invoke("vectorize_doc", { docId });
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

// Chat streaming callbacks

export type ChatStreamCallbacks = {
  onTextDelta?: (text: string) => void;
  onToolCallStart?: (name: string) => void;
  onToolCallDelta?: (text: string) => void;
  onToolCallDone?: (id: string, name: string, args: unknown) => void;
  onMessageDone?: (role: string, content: string | null, toolCalls: unknown | null) => void;
  onCitationData?: (items: EvidenceItem[], toolCallId: string, toolName: string) => void;
  onAgentDone?: () => void;
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
 *       onAgentDone: () => {
 *         // Agent finished — clean up listener
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
        callbacks.onMessageDone?.(ev.role, ev.content ?? null, ev.tool_calls ?? null);
        break;
      case "CitationData":
        callbacks.onCitationData?.(ev.items, ev.tool_call_id, ev.tool_name);
        break;
      case "AgentDone":
        callbacks.onAgentDone?.();
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

// Settings management

export async function readSettings(): Promise<ApiSettings> {
  return invoke("read_settings");
}

export async function saveSettings(settings: ApiSettings): Promise<void> {
  return invoke("save_settings", { settings });
}
