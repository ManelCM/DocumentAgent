"""DocumentAgent: OCR multimodal estructural con LangGraph."""

__all__ = ["build_graph"]


def build_graph():
    from .graph import build_graph as _build_graph
    return _build_graph()
