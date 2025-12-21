# PR Description: Advanced Ontology Extraction & Notebook Fixes

## 🚀 Summary
This PR revamps the `12_Unstructured_to_Ontology.ipynb` cookbook to resolve critical `TypeError` issues and significantly enhances the educational value of the ontology extraction guide. It also includes documentation updates and minor fixes across the context engineering module.

## 🐛 Fixes & Improvements

### 1. Fix: `TypeError: 'Entity' object is not subscriptable`
- **Issue**: The original notebook attempted to access `Entity` and `Relation` objects using dictionary syntax (e.g., `e['text']`), causing runtime errors because these are implemented as Python `dataclasses`.
- **Resolution**: Updated all access patterns to use dot notation (e.g., `e.text`) and added explicit `to_dict()` conversion logic before passing data to `OntologyGenerator`.

### 2. Feature: Enhanced Ontology Pipeline Guide
- **Classical NLP Pipeline**: detailed breakdown of using `NERExtractor` and `RelationExtractor` with proper object handling.
- **Generative AI Pipeline**: Added a robust example using `LLMOntologyGenerator` for direct schema generation from text.
- **Visualization**: Integrated `OntologyVisualizer` to compare outputs from both pipelines side-by-side.
- **Export**: Added steps to export the generated ontology to OWL/Turtle format (`.ttl`).

### 3. Documentation & Cleanup
- **`10_Temporal_Knowledge_Graphs.ipynb`**: Cleaned up dependencies (removed Docker setup in favor of `pip install semantica`).
- **`docs/reference/context.md`**: Fixed duplicate entries and added documentation for the production Graph Store.
- **`semantica/graph_store/graph_store.py`**: Fixed a `NameError` (missing `Tuple` import).

## 🛠️ Key Changes
- `cookbook/advanced/12_Unstructured_to_Ontology.ipynb`: **Complete Rewrite**
- `cookbook/advanced/10_Temporal_Knowledge_Graphs.ipynb`: **Updated**
- `docs/reference/context.md`: **Updated**
- `semantica/graph_store/graph_store.py`: **Fixed**

## ✅ Verification
- Validated that `NERExtractor` returns `Entity` objects and they are correctly processed.
- Verified that `OntologyGenerator` receives correctly formatted dictionaries.
- Ensured the notebook runs end-to-end without syntax errors.

## 📦 Dependencies
- No new external dependencies.
- Relies on existing `semantica` package structure.
