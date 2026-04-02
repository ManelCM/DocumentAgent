from __future__ import annotations

from .nodes import (
    node_aggregate,
    node_association,
    node_chart_specialist,
    node_cross_reference,
    node_detect_layout,
    node_formula_specialist,
    node_hierarchy,
    node_image_specialist,
    node_load_document,
    node_mixed_specialist,
    node_other_specialist,
    node_reduce_specialists,
    node_reading_order,
    node_table_specialist,
    node_text_specialist,
)
from .types import AgentState


def build_graph():
    try:
        from langgraph.graph import END, StateGraph

        graph = StateGraph(AgentState)

        # ── Sequential pipeline ──────────────────────────────────────────────
        graph.add_node("load_document", node_load_document)
        graph.add_node("detect_layout", node_detect_layout)
        graph.add_node("reading_order", node_reading_order)
        graph.add_node("hierarchy", node_hierarchy)          # includes inline math stitching

        # ── Parallel specialist fan-out ──────────────────────────────────────
        graph.add_node("text_specialist", node_text_specialist)
        graph.add_node("mixed_specialist", node_mixed_specialist)   # stitched inline math
        graph.add_node("image_specialist", node_image_specialist)
        graph.add_node("chart_specialist", node_chart_specialist)
        graph.add_node("formula_specialist", node_formula_specialist)
        graph.add_node("table_specialist", node_table_specialist)
        graph.add_node("other_specialist", node_other_specialist)

        # ── Reduction + post-processing ──────────────────────────────────────
        graph.add_node("reduce_specialists", node_reduce_specialists)
        graph.add_node("cross_reference", node_cross_reference)
        graph.add_node("association_node", node_association)
        graph.add_node("aggregate", node_aggregate)

        # ── Edges ────────────────────────────────────────────────────────────
        graph.set_entry_point("load_document")
        graph.add_edge("load_document", "detect_layout")
        graph.add_edge("detect_layout", "reading_order")
        graph.add_edge("reading_order", "hierarchy")

        # Fan-out: hierarchy → all specialists in parallel
        for specialist in [
            "text_specialist",
            "mixed_specialist",
            "image_specialist",
            "chart_specialist",
            "formula_specialist",
            "table_specialist",
            "other_specialist",
        ]:
            graph.add_edge("hierarchy", specialist)
            graph.add_edge(specialist, "reduce_specialists")

        graph.add_edge("reduce_specialists", "cross_reference")
        graph.add_edge("cross_reference", "association_node")
        graph.add_edge("association_node", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()

    except ModuleNotFoundError:
        # Fallback: run nodes sequentially without LangGraph
        class FallbackGraph:
            def invoke(self, state: AgentState) -> AgentState:
                for fn in [
                    node_load_document,
                    node_detect_layout,
                    node_reading_order,
                    node_hierarchy,
                    node_text_specialist,
                    node_mixed_specialist,
                    node_image_specialist,
                    node_chart_specialist,
                    node_formula_specialist,
                    node_table_specialist,
                    node_other_specialist,
                    node_reduce_specialists,
                    node_cross_reference,
                    node_association,
                    node_aggregate,
                ]:
                    state = fn(state)
                return state

        return FallbackGraph()
