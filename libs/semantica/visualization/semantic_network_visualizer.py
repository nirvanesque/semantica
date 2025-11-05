"""
Semantic Network Visualizer

This module provides visualization capabilities for semantic networks.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px

from ..utils.logging import get_logger
from ..utils.exceptions import ProcessingError
from .utils.layout_algorithms import ForceDirectedLayout
from .utils.color_schemes import ColorPalette, ColorScheme
from .utils.export_formats import export_plotly_figure


class SemanticNetworkVisualizer:
    """
    Semantic network visualizer.
    
    Provides visualization methods for semantic networks.
    """
    
    def __init__(self, **config):
        """Initialize semantic network visualizer."""
        self.logger = get_logger("semantic_network_visualizer")
        self.config = config
        color_scheme_name = config.get("color_scheme", "default")
        try:
            self.color_scheme = ColorScheme[color_scheme_name.upper()]
        except (KeyError, AttributeError):
            self.color_scheme = ColorScheme.DEFAULT
    
    def visualize_network(
        self,
        semantic_network: Any,
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize semantic network.
        
        Args:
            semantic_network: SemanticNetwork object
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing semantic network")
        
        # Extract nodes and edges from semantic network
        nodes = []
        if hasattr(semantic_network, "nodes"):
            for node in semantic_network.nodes:
                nodes.append({
                    "id": node.id,
                    "label": node.label,
                    "type": node.type,
                    "metadata": node.metadata
                })
        elif isinstance(semantic_network, dict):
            for node in semantic_network.get("nodes", []):
                nodes.append({
                    "id": node.get("id", ""),
                    "label": node.get("label", ""),
                    "type": node.get("type", ""),
                    "metadata": node.get("metadata", {})
                })
        
        edges = []
        if hasattr(semantic_network, "edges"):
            for edge in semantic_network.edges:
                edges.append({
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "metadata": edge.metadata
                })
        elif isinstance(semantic_network, dict):
            for edge in semantic_network.get("edges", []):
                edges.append({
                    "source": edge.get("source", ""),
                    "target": edge.get("target", ""),
                    "label": edge.get("label", ""),
                    "metadata": edge.get("metadata", {})
                })
        
        # Use KG visualizer for network visualization
        from .kg_visualizer import KGVisualizer
        graph = {"entities": nodes, "relationships": edges}
        kg_viz = KGVisualizer(**self.config)
        return kg_viz.visualize_network(graph, output, file_path, **options)
    
    def visualize_node_types(
        self,
        semantic_network: Any,
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize node type distribution.
        
        Args:
            semantic_network: SemanticNetwork object
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing semantic network node types")
        
        # Extract nodes
        nodes = []
        if hasattr(semantic_network, "nodes"):
            nodes = semantic_network.nodes
        elif isinstance(semantic_network, dict):
            nodes = semantic_network.get("nodes", [])
        
        # Count node types
        type_counts = {}
        for node in nodes:
            node_type = node.type if hasattr(node, "type") else node.get("type", "Unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        fig = px.bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            labels={"x": "Node Type", "y": "Count"},
            title="Semantic Network Node Type Distribution"
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def visualize_edge_types(
        self,
        semantic_network: Any,
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize edge type distribution.
        
        Args:
            semantic_network: SemanticNetwork object
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing semantic network edge types")
        
        # Extract edges
        edges = []
        if hasattr(semantic_network, "edges"):
            edges = semantic_network.edges
        elif isinstance(semantic_network, dict):
            edges = semantic_network.get("edges", [])
        
        # Count edge types
        type_counts = {}
        for edge in edges:
            edge_type = edge.label if hasattr(edge, "label") else edge.get("label", "Unknown")
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        
        fig = px.bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            labels={"x": "Edge Type", "y": "Count"},
            title="Semantic Network Edge Type Distribution"
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None

