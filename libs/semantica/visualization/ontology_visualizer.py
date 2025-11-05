"""
Ontology Visualizer

This module provides visualization capabilities for ontologies including
class hierarchies, property graphs, and ontology structure visualizations.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

try:
    import graphviz
except ImportError:
    graphviz = None

from ..utils.logging import get_logger
from ..utils.exceptions import ProcessingError
from .utils.layout_algorithms import HierarchicalLayout
from .utils.color_schemes import ColorPalette, ColorScheme
from .utils.export_formats import export_plotly_figure, export_matplotlib_figure, save_html


class OntologyVisualizer:
    """
    Ontology visualizer.
    
    Provides visualization methods for ontologies including:
    - Class hierarchy trees
    - Property graphs
    - Ontology structure networks
    - Class-property matrices
    """
    
    def __init__(self, **config):
        """
        Initialize ontology visualizer.
        
        Args:
            **config: Configuration options:
                - color_scheme: Color scheme name
                - node_size: Base node size
        """
        self.logger = get_logger("ontology_visualizer")
        self.config = config
        
        color_scheme_name = config.get("color_scheme", "default")
        try:
            self.color_scheme = ColorScheme[color_scheme_name.upper()]
        except (KeyError, AttributeError):
            self.color_scheme = ColorScheme.DEFAULT
        self.node_size = config.get("node_size", 15)
    
    def visualize_hierarchy(
        self,
        ontology: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize class hierarchy as tree.
        
        Args:
            ontology: Ontology dictionary with classes
            output: Output type ("interactive", "html", "png", "svg", "dot")
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing ontology class hierarchy")
        
        classes = ontology.get("classes", [])
        
        if not classes:
            raise ProcessingError("No classes found in ontology")
        
        # If output is dot and graphviz is available, use it
        if output == "dot" and graphviz is not None and file_path:
            return self._visualize_hierarchy_graphviz(classes, file_path, **options)
        
        # Build hierarchy tree
        hierarchy = self._build_hierarchy_tree(classes)
        
        return self._visualize_hierarchy_plotly(hierarchy, classes, output, file_path, **options)
    
    def visualize_properties(
        self,
        ontology: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize property graph showing properties and their domains/ranges.
        
        Args:
            ontology: Ontology dictionary
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing ontology properties")
        
        properties = ontology.get("properties", [])
        classes = ontology.get("classes", [])
        
        if not properties:
            raise ProcessingError("No properties found in ontology")
        
        return self._visualize_properties_plotly(properties, classes, output, file_path, **options)
    
    def visualize_structure(
        self,
        ontology: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize ontology structure as network.
        
        Args:
            ontology: Ontology dictionary
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing ontology structure")
        
        classes = ontology.get("classes", [])
        properties = ontology.get("properties", [])
        
        # Build nodes and edges
        nodes = []
        edges = []
        
        # Add class nodes
        for cls in classes:
            nodes.append({
                "id": cls.get("name") or cls.get("uri", ""),
                "label": cls.get("label") or cls.get("name", ""),
                "type": "class"
            })
            
            # Add hierarchy edges
            parent = cls.get("parent") or cls.get("subClassOf")
            if parent:
                edges.append({
                    "source": cls.get("name") or cls.get("uri", ""),
                    "target": parent,
                    "type": "subClassOf"
                })
        
        # Add property nodes
        for prop in properties:
            prop_name = prop.get("name") or prop.get("uri", "")
            nodes.append({
                "id": prop_name,
                "label": prop.get("label") or prop.get("name", ""),
                "type": "property"
            })
            
            # Add domain edges
            domain = prop.get("domain")
            if domain:
                edges.append({
                    "source": prop_name,
                    "target": domain,
                    "type": "domain"
                })
            
            # Add range edges
            range_val = prop.get("range")
            if range_val:
                edges.append({
                    "source": prop_name,
                    "target": range_val,
                    "type": "range"
                })
        
        return self._visualize_structure_plotly(nodes, edges, output, file_path, **options)
    
    def visualize_class_property_matrix(
        self,
        ontology: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize class-property matrix showing which properties belong to which classes.
        
        Args:
            ontology: Ontology dictionary
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing class-property matrix")
        
        classes = ontology.get("classes", [])
        properties = ontology.get("properties", [])
        
        # Build matrix
        class_names = [cls.get("name") or cls.get("label", "") for cls in classes]
        property_names = [prop.get("name") or prop.get("label", "") for prop in properties]
        
        matrix = []
        for cls in classes:
            row = []
            cls_props = cls.get("properties", [])
            for prop in properties:
                prop_name = prop.get("name") or prop.get("uri", "")
                # Check if property belongs to class (via domain or direct property list)
                domain = prop.get("domain")
                has_prop = prop_name in cls_props or domain == cls.get("name")
                row.append(1 if has_prop else 0)
            matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=property_names,
            y=class_names,
            colorscale='Blues',
            text=matrix,
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        fig.update_layout(
            title="Class-Property Matrix",
            xaxis_title="Properties",
            yaxis_title="Classes",
            width=800,
            height=600
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def visualize_metrics(
        self,
        ontology: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize ontology metrics dashboard.
        
        Args:
            ontology: Ontology dictionary
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing ontology metrics")
        
        classes = ontology.get("classes", [])
        properties = ontology.get("properties", [])
        
        # Calculate metrics
        num_classes = len(classes)
        num_properties = len(properties)
        
        # Calculate hierarchy depth
        max_depth = 0
        for cls in classes:
            depth = self._calculate_class_depth(cls, classes)
            max_depth = max(max_depth, depth)
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Class Count", "Property Count", "Max Depth"),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=num_classes,
            title={"text": "Classes"},
            domain={'x': [0, 0.33], 'y': [0, 1]}
        ), row=1, col=1)
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=num_properties,
            title={"text": "Properties"},
            domain={'x': [0.33, 0.66], 'y': [0, 1]}
        ), row=1, col=2)
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=max_depth,
            title={"text": "Max Depth"},
            domain={'x': [0.66, 1], 'y': [0, 1]}
        ), row=1, col=3)
        
        fig.update_layout(title="Ontology Metrics Dashboard")
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def _build_hierarchy_tree(self, classes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build hierarchy tree structure."""
        tree = {}
        root_classes = []
        
        for cls in classes:
            cls_name = cls.get("name") or cls.get("uri", "")
            parent = cls.get("parent") or cls.get("subClassOf")
            
            if parent:
                if parent not in tree:
                    tree[parent] = []
                tree[parent].append(cls_name)
            else:
                root_classes.append(cls_name)
            
            if cls_name not in tree:
                tree[cls_name] = []
        
        return tree
    
    def _calculate_class_depth(self, cls: Dict[str, Any], all_classes: List[Dict[str, Any]]) -> int:
        """Calculate depth of class in hierarchy."""
        parent = cls.get("parent") or cls.get("subClassOf")
        if not parent:
            return 1
        
        # Find parent class
        for p_cls in all_classes:
            if (p_cls.get("name") or p_cls.get("uri", "")) == parent:
                return 1 + self._calculate_class_depth(p_cls, all_classes)
        
        return 1
    
    def _visualize_hierarchy_plotly(
        self,
        hierarchy: Dict[str, List[str]],
        classes: List[Dict[str, Any]],
        output: str,
        file_path: Optional[Path],
        **options
    ) -> Optional[Any]:
        """Create Plotly hierarchy visualization."""
        # Build tree structure for plotly
        # This is a simplified tree visualization
        # For full tree, would need more complex layout
        
        # Get root classes
        all_class_names = {cls.get("name") or cls.get("uri", ""): cls for cls in classes}
        root_classes = [name for name in all_class_names.keys() if name not in [c for children in hierarchy.values() for c in children]]
        
        if not root_classes:
            root_classes = list(all_class_names.keys())[:1] if all_class_names else []
        
        # Build node and edge lists for tree
        nodes = []
        edges = []
        
        def add_node_and_children(cls_name, level=0, x_offset=0):
            nodes.append({
                "name": cls_name,
                "level": level,
                "x": x_offset,
                "y": -level
            })
            
            children = hierarchy.get(cls_name, [])
            child_width = 1.0 / max(len(children), 1)
            for i, child in enumerate(children):
                child_x = x_offset - 0.5 + (i + 0.5) * child_width
                edges.append({
                    "source": cls_name,
                    "target": child
                })
                add_node_and_children(child, level + 1, child_x)
        
        # Start from root
        root_x = 0.5
        root_width = 1.0 / max(len(root_classes), 1)
        for i, root in enumerate(root_classes):
            root_x = (i + 0.5) * root_width
            add_node_and_children(root, 0, root_x)
        
        # Create visualization
        edge_x = []
        edge_y = []
        node_positions = {n["name"]: (n["x"], n["y"]) for n in nodes}
        
        for edge in edges:
            source_pos = node_positions.get(edge["source"])
            target_pos = node_positions.get(edge["target"])
            if source_pos and target_pos:
                edge_x.extend([source_pos[0], target_pos[0], None])
                edge_y.extend([source_pos[1], target_pos[1], None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = [n["x"] for n in nodes]
        node_y = [n["y"] for n in nodes]
        node_text = [n["name"] for n in nodes]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            marker=dict(size=self.node_size*10, color='lightblue', line=dict(width=2, color='darkblue'))
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Ontology Class Hierarchy',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def _visualize_hierarchy_graphviz(
        self,
        classes: List[Dict[str, Any]],
        file_path: Path,
        **options
    ) -> None:
        """Create Graphviz hierarchy visualization."""
        if graphviz is None:
            raise ProcessingError("Graphviz not available. Install with: pip install graphviz")
        
        dot = graphviz.Digraph(comment='Ontology Hierarchy')
        
        # Add nodes
        for cls in classes:
            cls_name = cls.get("name") or cls.get("label", "")
            dot.node(cls_name, cls_name)
        
        # Add edges
        for cls in classes:
            cls_name = cls.get("name") or cls.get("label", "")
            parent = cls.get("parent") or cls.get("subClassOf")
            if parent:
                dot.edge(parent, cls_name)
        
        dot.render(str(file_path), format='svg', cleanup=True)
        self.logger.info(f"Saved Graphviz hierarchy to {file_path}")
    
    def _visualize_properties_plotly(
        self,
        properties: List[Dict[str, Any]],
        classes: List[Dict[str, Any]],
        output: str,
        file_path: Optional[Path],
        **options
    ) -> Optional[Any]:
        """Create Plotly property visualization."""
        # Build property graph
        nodes = []
        edges = []
        
        # Add class nodes
        for cls in classes:
            cls_name = cls.get("name") or cls.get("uri", "")
            nodes.append({"id": cls_name, "label": cls.get("label") or cls_name, "type": "class"})
        
        # Add property nodes and edges
        for prop in properties:
            prop_name = prop.get("name") or prop.get("uri", "")
            nodes.append({"id": prop_name, "label": prop.get("label") or prop_name, "type": "property"})
            
            domain = prop.get("domain")
            if domain:
                edges.append({"source": prop_name, "target": domain, "type": "domain"})
            
            range_val = prop.get("range")
            if range_val:
                edges.append({"source": prop_name, "target": range_val, "type": "range"})
        
        # Use similar approach as KG visualizer
        from .kg_visualizer import KGVisualizer
        kg_viz = KGVisualizer(**self.config)
        return kg_viz._visualize_network_plotly(nodes, edges, output, file_path, **options)
    
    def _visualize_structure_plotly(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        output: str,
        file_path: Optional[Path],
        **options
    ) -> Optional[Any]:
        """Create Plotly structure visualization."""
        from .kg_visualizer import KGVisualizer
        kg_viz = KGVisualizer(**self.config)
        return kg_viz._visualize_network_plotly(nodes, edges, output, file_path, **options)

