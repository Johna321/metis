from dataclasses import dataclass
from typing import List, Tuple, Optional

BBox = Tuple[float, float, float, float]  # x0,y0,x1,y1

@dataclass(frozen=True)
class Span:
    span_id: str
    doc_id: str
    page: int               # 0-index
    bbox_pdf: BBox          # points, PyMuPDF coords (top-left origin)
    bbox_norm: BBox         # normalized [0,1] top-left origin
    text: str
    reading_order: int
    is_header: bool = False
    is_footer: bool = False
    # --- layout-engine fields ---
    kind: Optional[str] = None          # e.g. "text", "table", "picture", ...
    pos: Optional[Tuple[int, int]] = None  # char offsets into per-page markdown
    source: str = "pymupdf_blocks"      # "pymupdf_blocks" | "pymupdf4llm_page_boxes" | "pymupdf4llm_layout"

@dataclass(frozen=True)
class Evidence:
    span_id: str
    page: int
    bbox_norm: BBox
    text: str
    score: float

# --- Agent message types ---

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str

@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None

