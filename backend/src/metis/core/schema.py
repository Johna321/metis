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

@dataclass(frozen=True)
class Evidence:
    span_id: str
    page: int
    bbox_norm: BBox
    text: str
    score: float

