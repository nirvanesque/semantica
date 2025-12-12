# Refactor: Rename `triple_store` to `triplet_store`

## Summary
This Pull Request renames the `semantica/triple_store` module to `semantica/triplet_store` and updates all references across the codebase to ensure consistent naming conventions.

## Motivation
The term "triplet" is the standard terminology used within the Semantica framework. This refactor aligns the module name, class names, and documentation with this convention, eliminating ambiguity and "triple"/"triplet" inconsistencies.

## Changes
- **Module Rename**: Renamed directory `semantica/triple_store` -> `semantica/triplet_store`.
- **Core Updates**:
  - Updated `__init__.py`, `triplet_manager.py`, `query_engine.py`, `bulk_loader.py`, and all adapters (`blazegraph`, `jena`, `rdf4j`, `virtuoso`) to use `triplet_store` imports.
  - Renamed classes: `TripleManager` -> `TripletManager`, `TripleStore` -> `TripletStore`.
- **Notebook Refactoring**:
  - Updated imports and usage in `cookbook/introduction/20_Triplet_Store.ipynb`.
  - Updated `cookbook/advanced/09_Semantic_Layer_Construction.ipynb`.
  - Updated healthcare use cases: `01_Clinical_Reports_Processing.ipynb`, `04_Healthcare_GraphRAG_Hybrid.ipynb`, `05_Medical_Database_Integration.ipynb`, `06_Patient_Records_Temporal.ipynb`.
- **Documentation**:
  - Renamed `docs/reference/triple_store.md` -> `triplet_store.md`.
  - Updated `README.md`, `docs/modules.md`, `docs/glossary.md`, `docs/CodeExamples.md`, `docs/reference/graph_store.md`, `docs/reference/reasoning.md`.
  - Updated `mkdocs.yml` navigation.
- **Tests**:
  - Renamed `tests/triple_store` -> `tests/triplet_store`.
  - Updated `test_triplet_store.py` to test the renamed module.

## Verification
- **Tests**: Ran `pytest tests/triplet_store/test_triplet_store.py`. All tests passed.
- **Static Analysis**: Verified no lingering `semantica.triple_store` imports remain in the codebase (grep check).

## Breaking Changes
- `semantica.triple_store` is no longer available. Users must update imports to `semantica.triplet_store`.
- `TripleManager` and `TripleStore` classes are renamed to `TripletManager` and `TripletStore`.
