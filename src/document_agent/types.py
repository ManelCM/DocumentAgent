from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict


class BlockType(str, Enum):
    TEXT = "text"
    FORMULA = "formula"
    IMAGE = "image"
    CHART = "chart"
    FIGURE = "figure"
    CAPTION = "caption"
    OTHER = "other"


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class DocumentBlock:
    block_id: str
    page_index: int
    bbox: BoundingBox
    block_type: BlockType
    detector_label: str
    confidence: float
    reading_order: int = -1
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    relations: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.block_id,
            "page_index": self.page_index,
            "bbox": {
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "x2": self.bbox.x2,
                "y2": self.bbox.y2,
            },
            "type": self.block_type.value,
            "detector_label": self.detector_label,
            "confidence": float(self.confidence),
            "reading_order": int(self.reading_order),
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "relations": self.relations,
            "payload": self.payload,
        }


class AgentState(TypedDict, total=False):
    input_path: str
    run_id: str
    pages: List[Any]
    page_sizes: List[Dict[str, int]]
    blocks: List[DocumentBlock]
    warnings: List[str]
    text_updates: Dict[str, Dict[str, Any]]
    image_updates: Dict[str, Dict[str, Any]]
    chart_updates: Dict[str, Dict[str, Any]]
    formula_updates: Dict[str, Dict[str, Any]]
    other_updates: Dict[str, Dict[str, Any]]
    text_warnings: List[str]
    image_warnings: List[str]
    chart_warnings: List[str]
    formula_warnings: List[str]
    other_warnings: List[str]
    output: Dict[str, Any]
    status: Literal["init", "running", "done", "error"]
