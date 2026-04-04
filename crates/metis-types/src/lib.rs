use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IngestResponse {
    pub doc_id: String,
    pub n_pages: u32,
    pub n_spans: u32,
    pub ingest: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VectorizeResponse {
    pub doc_id: String,
    pub n_embedded: u32,
    pub n_skipped: Option<u32>,
    pub model: String,
    pub dim: Option<u32>,
    pub was_cached: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EvidenceItem {
    pub span_id: String,
    pub page: u32,
    pub bbox_norm: [f64; 4],
    pub text: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BboxSelection {
    pub page: u32,
    pub bbox_norm: [f64; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConversationMeta {
    pub id: String,
    pub title: String,
    pub pinned: bool,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
    pub evidence: Option<Vec<EvidenceItem>>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConversationFull {
    pub id: String,
    pub title: String,
    pub pinned: bool,
    pub messages: Vec<ConversationMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChatRequest {
    pub doc_id: String,
    pub message: String,
    pub conv_id: Option<String>,
    pub selections: Option<Vec<BboxSelection>>,
    pub provider: Option<String>,
    pub model: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind")]
pub enum ChatStreamEvent {
    TextDelta {
        text: String,
    },
    ToolCallStart {
        name: String,
    },
    ToolCallDelta {
        text: String,
    },
    ToolCallDone {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    MessageDone {
        role: String,
        content: Option<String>,
        tool_calls: Option<serde_json::Value>,
    },
    CitationData {
        tool_call_id: String,
        tool_name: String,
        items: Vec<EvidenceItem>,
    },
    TitleUpdate {
        conv_id: String,
        title: String,
    },
    AgentDone,
    Error {
        message: String,
    },
}
