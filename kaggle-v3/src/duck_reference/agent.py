"""Stock Duck ToolAgent selected through an isolated import path."""

from inference.agent.tool_agent import ToolAgent


class DuckReferenceToolAgent(ToolAgent):
    """Marker type for the audited, hybrid-disabled Duck behavior."""
