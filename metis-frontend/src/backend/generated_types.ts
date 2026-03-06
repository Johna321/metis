/* eslint-disable */
/**
 * AUTO-GENERATED — do not edit by hand.
 * Source: crates/metis-types/src/lib.rs
 * Run ./scripts/sync-types.sh to regenerate.
 */


export interface BboxSelection {
  bbox_norm: [number, number, number, number];
  page: number;
}


export interface IngestResponse {
  doc_id: string;
  ingest: unknown;
  n_pages: number;
  n_spans: number;
}


export interface VectorizeResponse {
  dim?: number | null;
  doc_id: string;
  model: string;
  n_embedded: number;
  n_skipped?: number | null;
  was_cached: boolean;
}


export interface EvidenceItem {
  bbox_norm: [number, number, number, number];
  page: number;
  score: number;
  span_id: string;
  text: string;
}


export interface ChatRequest {
  doc_id: string;
  message: string;
  model?: string | null;
  provider?: string | null;
  selections?: BboxSelection[] | null;
}


export type ChatStreamEvent =
  | {
      kind: "TextDelta";
      text: string;
    }
  | {
      kind: "ToolCallStart";
      name: string;
    }
  | {
      kind: "ToolCallDelta";
      text: string;
    }
  | {
      arguments: unknown;
      id: string;
      kind: "ToolCallDone";
      name: string;
    }
  | {
      content?: string | null;
      kind: "MessageDone";
      role: string;
      tool_calls?: unknown;
    }
  | {
      items: EvidenceItem[];
      kind: "CitationData";
      tool_call_id: string;
      tool_name: string;
    }
  | {
      kind: "AgentDone";
    }
  | {
      kind: "Error";
      message: string;
    };

