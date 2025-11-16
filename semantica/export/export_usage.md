# Export Module Usage Guide

This guide demonstrates how to use the export module for exporting knowledge graphs, entities, relationships, and data to various formats including RDF, JSON, CSV, Graph, YAML, OWL, Vector, and LPG formats.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [RDF Export](#rdf-export)
3. [JSON/JSON-LD Export](#jsonjson-ld-export)
4. [CSV Export](#csv-export)
5. [Graph Export](#graph-export)
6. [YAML Export](#yaml-export)
7. [OWL Export](#owl-export)
8. [Vector Export](#vector-export)
9. [LPG Export](#lpg-export)
10. [Report Generation](#report-generation)
11. [Knowledge Graph Export](#knowledge-graph-export)
12. [Using Methods](#using-methods)
13. [Using Registry](#using-registry)
14. [Configuration](#configuration)
15. [Advanced Examples](#advanced-examples)

## Basic Usage

### Using the Convenience Function

```python
from semantica.export import export_knowledge_graph

# Export knowledge graph to JSON
kg = {
    "entities": [...],
    "relationships": [...],
    "metadata": {...}
}

export_knowledge_graph(kg, "output.json", format="json")
export_knowledge_graph(kg, "output.ttl", format="turtle")
export_knowledge_graph(kg, "output.cypher", format="cypher")
```

### Using Main Classes

```python
from semantica.export import JSONExporter, RDFExporter, LPGExporter

# Create exporters
json_exporter = JSONExporter(indent=2)
rdf_exporter = RDFExporter()
lpg_exporter = LPGExporter()

# Export knowledge graph
json_exporter.export_knowledge_graph(kg, "output.json")
rdf_exporter.export(kg, "output.ttl", format="turtle")
lpg_exporter.export_knowledge_graph(kg, "output.cypher")
```

## RDF Export

### Turtle Format

```python
from semantica.export import RDFExporter

exporter = RDFExporter()

# Export to Turtle format
exporter.export(kg, "output.ttl", format="turtle")

# Export knowledge graph
exporter.export_knowledge_graph(kg, "kg.ttl", format="turtle")
```

### RDF/XML Format

```python
from semantica.export import RDFExporter

exporter = RDFExporter()

# Export to RDF/XML format
exporter.export(kg, "output.rdf", format="rdfxml")
```

### JSON-LD Format

```python
from semantica.export import RDFExporter

exporter = RDFExporter()

# Export to JSON-LD format
exporter.export(kg, "output.jsonld", format="jsonld")
```

### Using RDF Export Methods

```python
from semantica.export.methods import export_rdf

# Turtle format
export_rdf(kg, "output.ttl", format="turtle")

# RDF/XML format
export_rdf(kg, "output.rdf", format="rdfxml")

# JSON-LD format
export_rdf(kg, "output.jsonld", format="jsonld")
```

## JSON/JSON-LD Export

### Standard JSON Export

```python
from semantica.export import JSONExporter

exporter = JSONExporter(
    indent=2,
    ensure_ascii=False,
    format="json"
)

# Export knowledge graph
exporter.export_knowledge_graph(kg, "output.json")

# Export entities
exporter.export_entities(entities, "entities.json")
```

### JSON-LD Export

```python
from semantica.export import JSONExporter

exporter = JSONExporter(format="json-ld")

# Export knowledge graph as JSON-LD
exporter.export_knowledge_graph(kg, "output.jsonld", format="json-ld")
```

### Using JSON Export Methods

```python
from semantica.export.methods import export_json

# Standard JSON
export_json(kg, "output.json", format="json")

# JSON-LD
export_json(kg, "output.jsonld", format="json-ld")
```

## CSV Export

### Entity Export

```python
from semantica.export import CSVExporter

exporter = CSVExporter(
    delimiter=",",
    encoding="utf-8",
    include_header=True
)

# Export entities
exporter.export_entities(entities, "entities.csv")

# Export relationships
exporter.export_relationships(relationships, "relationships.csv")
```

### Knowledge Graph Export

```python
from semantica.export import CSVExporter

exporter = CSVExporter()

# Export knowledge graph to multiple CSV files
exporter.export_knowledge_graph(kg, "kg_base")
# Creates: kg_base_entities.csv, kg_base_relationships.csv
```

### Using CSV Export Methods

```python
from semantica.export.methods import export_csv

# Export entities
export_csv(entities, "entities.csv")

# Export knowledge graph
export_csv(kg, "kg_base")
```

## Graph Export

### GraphML Format

```python
from semantica.export import GraphExporter

exporter = GraphExporter(
    format="graphml",
    include_attributes=True
)

# Export knowledge graph
exporter.export_knowledge_graph(kg, "graph.graphml")
```

### GEXF Format

```python
from semantica.export import GraphExporter

exporter = GraphExporter(format="gexf")

# Export knowledge graph
exporter.export_knowledge_graph(kg, "graph.gexf", format="gexf")
```

### DOT Format

```python
from semantica.export import GraphExporter

exporter = GraphExporter(format="dot")

# Export knowledge graph
exporter.export_knowledge_graph(kg, "graph.dot", format="dot")
```

### Using Graph Export Methods

```python
from semantica.export.methods import export_graph

# GraphML
export_graph(graph_data, "graph.graphml", format="graphml")

# GEXF
export_graph(graph_data, "graph.gexf", format="gexf")

# DOT
export_graph(graph_data, "graph.dot", format="dot")
```

## YAML Export

### Semantic Network Export

```python
from semantica.export import SemanticNetworkYAMLExporter

exporter = SemanticNetworkYAMLExporter()

# Export semantic network
exporter.export(semantic_network, "network.yaml")
```

### Schema Export

```python
from semantica.export import YAMLSchemaExporter

exporter = YAMLSchemaExporter()

# Export schema
exporter.export(schema, "schema.yaml")
```

### Using YAML Export Methods

```python
from semantica.export.methods import export_yaml

# Semantic network
export_yaml(semantic_network, "network.yaml", method="semantic_network")

# Schema
export_yaml(schema, "schema.yaml", method="schema")
```

## OWL Export

### OWL/XML Format

```python
from semantica.export import OWLExporter

exporter = OWLExporter(
    ontology_uri="http://example.org/ontology#",
    version="1.0",
    format="owl-xml"
)

# Export ontology
exporter.export(ontology, "ontology.owl", format="owl-xml")
```

### Turtle Format

```python
from semantica.export import OWLExporter

exporter = OWLExporter(format="turtle")

# Export ontology in Turtle format
exporter.export(ontology, "ontology.ttl", format="turtle")
```

### Using OWL Export Methods

```python
from semantica.export.methods import export_owl

# OWL/XML
export_owl(ontology, "ontology.owl", format="owl-xml")

# Turtle
export_owl(ontology, "ontology.ttl", format="turtle")
```

## Vector Export

### JSON Format

```python
from semantica.export import VectorExporter

exporter = VectorExporter(
    format="json",
    include_metadata=True,
    include_text=True
)

# Export vectors
exporter.export(vectors, "vectors.json")
```

### NumPy Format

```python
from semantica.export import VectorExporter

exporter = VectorExporter(format="numpy")

# Export vectors
exporter.export(vectors, "vectors.npy", format="numpy")
```

### FAISS Format

```python
from semantica.export import VectorExporter

exporter = VectorExporter(format="faiss")

# Export vectors for FAISS
exporter.export(vectors, "vectors.faiss", format="faiss")
```

### Using Vector Export Methods

```python
from semantica.export.methods import export_vector

# JSON
export_vector(vectors, "vectors.json", format="json")

# NumPy
export_vector(vectors, "vectors.npy", format="numpy")

# FAISS
export_vector(vectors, "vectors.faiss", format="faiss")
```

## LPG Export

### Cypher Query Export

```python
from semantica.export import LPGExporter

exporter = LPGExporter(
    batch_size=1000,
    include_indexes=True
)

# Export knowledge graph to Cypher queries
exporter.export_knowledge_graph(kg, "graph.cypher")
```

### Using LPG Export Methods

```python
from semantica.export.methods import export_lpg

# Export to Cypher format
export_lpg(kg, "graph.cypher", method="cypher")
```

### Importing into Neo4j

```python
from semantica.export import LPGExporter

exporter = LPGExporter()

# Export to Cypher file
exporter.export_knowledge_graph(kg, "graph.cypher")

# Then import into Neo4j using cypher-shell:
# cypher-shell -u neo4j -p password < graph.cypher
```

### Importing into Memgraph

```python
from semantica.export import LPGExporter

exporter = LPGExporter()

# Export to Cypher file
exporter.export_knowledge_graph(kg, "graph.cypher")

# Then import into Memgraph using mgconsole:
# mgconsole < graph.cypher
```

## Report Generation

### HTML Report

```python
from semantica.export import ReportGenerator

generator = ReportGenerator(
    format="html",
    include_charts=True
)

# Generate report
generator.generate_report(data, "report.html", format="html")
```

### Markdown Report

```python
from semantica.export import ReportGenerator

generator = ReportGenerator(format="markdown")

# Generate report
generator.generate_report(data, "report.md", format="markdown")
```

### Quality Report

```python
from semantica.export import ReportGenerator

generator = ReportGenerator()

# Generate quality report
generator.generate_quality_report(metrics, "quality.md", format="markdown")
```

### Using Report Generation Methods

```python
from semantica.export.methods import generate_report

# HTML report
generate_report(data, "report.html", format="html")

# Markdown report
generate_report(data, "report.md", format="markdown")
```

## Knowledge Graph Export

### Unified Export Function

```python
from semantica.export.methods import export_knowledge_graph

kg = {
    "entities": [...],
    "relationships": [...],
    "metadata": {...}
}

# Auto-detect format from file extension
export_knowledge_graph(kg, "output.json")  # JSON format
export_knowledge_graph(kg, "output.ttl")    # Turtle format
export_knowledge_graph(kg, "output.cypher") # Cypher format

# Explicit format specification
export_knowledge_graph(kg, "output.json", format="json-ld")
export_knowledge_graph(kg, "output.rdf", format="rdfxml")
```

### Multiple Format Export

```python
from semantica.export.methods import export_knowledge_graph

kg = {...}

# Export to multiple formats
formats = ["json", "turtle", "csv", "cypher"]
for fmt in formats:
    export_knowledge_graph(kg, f"output.{fmt}", format=fmt)
```

## Using Methods

### Format-Specific Methods

```python
from semantica.export.methods import (
    export_rdf,
    export_json,
    export_csv,
    export_graph,
    export_yaml,
    export_owl,
    export_vector,
    export_lpg
)

# RDF export
export_rdf(kg, "output.ttl", format="turtle")

# JSON export
export_json(kg, "output.json", format="json")

# CSV export
export_csv(entities, "entities.csv")

# Graph export
export_graph(graph_data, "graph.graphml", format="graphml")

# YAML export
export_yaml(semantic_network, "network.yaml")

# OWL export
export_owl(ontology, "ontology.owl", format="owl-xml")

# Vector export
export_vector(vectors, "vectors.json", format="json")

# LPG export
export_lpg(kg, "graph.cypher", method="cypher")
```

## Using Registry

### Registering Custom Methods

```python
from semantica.export.registry import method_registry

def custom_json_export(data, file_path, **kwargs):
    """Custom JSON export function."""
    # Your custom implementation
    pass

# Register custom method
method_registry.register("json", "custom_method", custom_json_export)

# Use custom method
from semantica.export.methods import export_json
export_json(data, "output.json", method="custom_method")
```

### Listing Available Methods

```python
from semantica.export.methods import list_available_methods

# List all methods
all_methods = list_available_methods()
print(all_methods)

# List methods for specific task
json_methods = list_available_methods("json")
print(json_methods)
```

### Getting Registered Methods

```python
from semantica.export.methods import get_export_method

# Get method from registry
method = get_export_method("json", "custom_method")
if method:
    method(data, "output.json")
```

## Configuration

### Environment Variables

```bash
# Set export configuration via environment variables
export EXPORT_DEFAULT_FORMAT="json"
export EXPORT_OUTPUT_DIR="./exports"
export EXPORT_INCLUDE_METADATA="true"
export EXPORT_INDENT=2
export EXPORT_ENCODING="utf-8"
export EXPORT_DELIMITER=","
export EXPORT_NAMESPACE_BASE="http://example.org/ns#"
export EXPORT_VALIDATE="true"
```

### Programmatic Configuration

```python
from semantica.export.config import export_config

# Set configuration
export_config.set("default_format", "json")
export_config.set("output_dir", "./exports")
export_config.set("include_metadata", True)

# Get configuration
format = export_config.get("default_format", default="json")
output_dir = export_config.get("output_dir", default="./")

# Method-specific configuration
export_config.set_method_config("rdf", format="turtle")
rdf_config = export_config.get_method_config("rdf")
```

### Config File (YAML)

```yaml
export:
  default_format: "json"
  output_dir: "./exports"
  include_metadata: true
  indent: 2
  encoding: "utf-8"
  delimiter: ","
  namespace_base: "http://example.org/ns#"
  validate: true

export_methods:
  rdf:
    format: "turtle"
    validate: true
  json:
    indent: 2
    ensure_ascii: false
  csv:
    delimiter: ","
    include_header: true
  lpg:
    batch_size: 1000
    include_indexes: true
```

```python
from semantica.export.config import ExportConfig

# Load from config file
config = ExportConfig(config_file="config.yaml")
```

## Advanced Examples

### Complete Export Pipeline

```python
from semantica.export.methods import (
    export_knowledge_graph,
    export_rdf,
    export_lpg
)

kg = {
    "entities": [...],
    "relationships": [...],
    "metadata": {...}
}

# Export to multiple formats
export_knowledge_graph(kg, "output.json", format="json")
export_rdf(kg, "output.ttl", format="turtle")
export_lpg(kg, "output.cypher", method="cypher")
```

### Batch Export with Format Detection

```python
from semantica.export.methods import export_knowledge_graph
from pathlib import Path

kg = {...}

# Export to multiple formats based on file extensions
output_files = [
    "output.json",
    "output.ttl",
    "output.csv",
    "output.cypher",
    "output.graphml"
]

for file_path in output_files:
    export_knowledge_graph(kg, file_path)
    # Format auto-detected from extension
```

### Custom Export Method

```python
from semantica.export.registry import method_registry
from semantica.export.methods import export_json

def custom_jsonld_export(data, file_path, **kwargs):
    """Custom JSON-LD export with specific context."""
    from semantica.export import JSONExporter
    
    exporter = JSONExporter(format="json-ld")
    
    # Add custom context
    if isinstance(data, dict) and "@context" not in data:
        data["@context"] = {
            "@vocab": "http://example.org/vocab#",
            "ex": "http://example.org/ns#"
        }
    
    exporter.export(data, file_path, format="json-ld", **kwargs)

# Register custom method
method_registry.register("json", "custom_jsonld", custom_jsonld_export)

# Use custom method
export_json(kg, "output.jsonld", method="custom_jsonld")
```

### LPG Export with Custom Configuration

```python
from semantica.export import LPGExporter

# Custom LPG exporter configuration
exporter = LPGExporter(
    batch_size=5000,  # Larger batches for big graphs
    include_indexes=True
)

# Export large knowledge graph
exporter.export_knowledge_graph(large_kg, "large_graph.cypher")
```

### Multi-Format Knowledge Graph Export

```python
from semantica.export.methods import export_knowledge_graph

kg = {...}

# Export to all supported formats
formats = {
    "json": "output.json",
    "json-ld": "output.jsonld",
    "turtle": "output.ttl",
    "rdfxml": "output.rdf",
    "csv": "output_base",
    "graphml": "graph.graphml",
    "cypher": "graph.cypher",
    "yaml": "network.yaml",
    "owl": "ontology.owl"
}

for format_name, file_path in formats.items():
    try:
        export_knowledge_graph(kg, file_path, format=format_name)
        print(f"✓ Exported to {format_name}: {file_path}")
    except Exception as e:
        print(f"✗ Failed to export {format_name}: {e}")
```

## Best Practices

1. **Format Selection**: Choose the appropriate format for your use case
   - JSON/JSON-LD: Web applications, APIs
   - RDF/Turtle: Semantic web, linked data
   - CSV: Tabular analysis, spreadsheets
   - GraphML/GEXF: Graph visualization tools
   - Cypher/LPG: Graph databases (Neo4j, Memgraph)
   - YAML: Human-readable, pipeline integration
   - OWL: Ontology modeling

2. **Batch Processing**: Use batch export for large knowledge graphs
   ```python
   exporter = LPGExporter(batch_size=5000)
   exporter.export_knowledge_graph(large_kg, "output.cypher")
   ```

3. **Metadata Preservation**: Include metadata for provenance tracking
   ```python
   export_config.set("include_metadata", True)
   ```

4. **Validation**: Enable validation for RDF/OWL exports
   ```python
   export_config.set("validate", True)
   ```

5. **Error Handling**: Always handle export errors gracefully
   ```python
   try:
       export_knowledge_graph(kg, "output.json")
   except Exception as e:
       logger.error(f"Export failed: {e}")
   ```

6. **Format Auto-Detection**: Use file extensions for automatic format detection
   ```python
   export_knowledge_graph(kg, "output.ttl")  # Auto-detects Turtle format
   ```

7. **Configuration Management**: Use environment variables or config files for consistent settings
   ```python
   export_config.set("default_format", "json")
   ```

## Performance Tips

1. **Batch Size**: Adjust batch size based on graph size and memory
   ```python
   exporter = LPGExporter(batch_size=10000)  # For very large graphs
   ```

2. **Parallel Export**: Export to multiple formats in parallel
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor() as executor:
       executor.submit(export_json, kg, "output.json")
       executor.submit(export_rdf, kg, "output.ttl", format="turtle")
       executor.submit(export_lpg, kg, "output.cypher")
   ```

3. **Streaming Export**: For very large graphs, consider streaming export
   ```python
   # Use batch processing for large exports
   exporter = CSVExporter()
   exporter.export_knowledge_graph(large_kg, "output_base")
   ```

4. **Caching**: Cache exported files when possible
   ```python
   # Check if export already exists
   if not Path("output.json").exists():
       export_knowledge_graph(kg, "output.json")
   ```

5. **Compression**: Compress large exports
   ```python
   import gzip
   export_knowledge_graph(kg, "output.json")
   # Then compress
   with open("output.json", "rb") as f_in:
       with gzip.open("output.json.gz", "wb") as f_out:
           f_out.writelines(f_in)
   ```

