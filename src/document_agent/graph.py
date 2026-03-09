from __future__ import annotations

from .nodes import (
    node_aggregate,
    node_association,
    node_chart_specialist,
    node_detect_layout,
    node_formula_specialist,
    node_hierarchy,
    node_image_specialist,
    node_load_document,
    node_other_specialist,
    node_reduce_specialists,
    node_reading_order,
    node_text_specialist,
)
from .types import AgentState


def build_graph():
    try:
        from langgraph.graph import END, StateGraph

        graph = StateGraph(AgentState)
        graph.add_node("load_document", node_load_document)
        graph.add_node("detect_layout", node_detect_layout)
        graph.add_node("reading_order", node_reading_order)
        graph.add_node("hierarchy", node_hierarchy)
        graph.add_node("text_specialist", node_text_specialist)
        graph.add_node("image_specialist", node_image_specialist)
        graph.add_node("chart_specialist", node_chart_specialist)
        graph.add_node("formula_specialist", node_formula_specialist)
        graph.add_node("other_specialist", node_other_specialist)
        graph.add_node("reduce_specialists", node_reduce_specialists)
        graph.add_node("association_node", node_association)
        graph.add_node("aggregate", node_aggregate)

        graph.set_entry_point("load_document")
        graph.add_edge("load_document", "detect_layout")
        graph.add_edge("detect_layout", "reading_order")
        graph.add_edge("reading_order", "hierarchy")
        graph.add_edge("hierarchy", "text_specialist")
        graph.add_edge("hierarchy", "image_specialist")
        graph.add_edge("hierarchy", "chart_specialist")
        graph.add_edge("hierarchy", "formula_specialist")
        graph.add_edge("hierarchy", "other_specialist")
        graph.add_edge("text_specialist", "reduce_specialists")
        graph.add_edge("image_specialist", "reduce_specialists")
        graph.add_edge("chart_specialist", "reduce_specialists")
        graph.add_edge("formula_specialist", "reduce_specialists")
        graph.add_edge("other_specialist", "reduce_specialists")
        graph.add_edge("reduce_specialists", "association_node")
        graph.add_edge("association_node", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()
    except ModuleNotFoundError:
        class FallbackGraph:
            def invoke(self, state: AgentState) -> AgentState:
                for fn in [
                    node_load_document,
                    node_detect_layout,
                    node_reading_order,
                    node_hierarchy,
                    node_text_specialist,
                    node_image_specialist,
                    node_chart_specialist,
                    node_formula_specialist,
                    node_other_specialist,
                    node_reduce_specialists,
                    node_association,
                    node_aggregate,
                ]:
                    state = fn(state)
                return state

        return FallbackGraph()
