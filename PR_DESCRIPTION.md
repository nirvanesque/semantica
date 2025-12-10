# PR: Dynamic Embedding Model Switching & Enhanced Testing

## Summary
This PR introduces dynamic model switching capabilities to the Semantica embeddings module, allowing users to switch between `sentence_transformers` and `fastembed` providers at runtime. It also includes comprehensive updates to documentation, cookbooks, and a robust new test suite to validate all core features.

## Key Changes

### 1. Dynamic Model Switching
- **Added** `set_model()` and `set_text_model()` methods to `EmbeddingGenerator` and `TextEmbedder`.
- **Feature**: Users can now change the underlying embedding model and provider without re-instantiating the classes.
- **Validation**: Added checks to ensure the requested provider (e.g., `fastembed`) is installed before switching.

### 2. Documentation Updates
- **`semantica/embeddings/embeddings_usage.md`**: Added a new section **"Dynamic Model Switching"** with code examples.
- **`docs/reference/embeddings.md`**: Updated API reference tables to include the new model selection methods.

### 3. Cookbook Refactoring
- **`12_Embedding_Generation.ipynb`**: Added **"Step 3: Model Selection & Dynamic Switching"** to demonstrate the new API.
- **`13_Vector_Store.ipynb`**: Replaced random vector generation with **real embeddings** using `TextEmbedder`, ensuring the tutorial reflects real-world usage.
- **`Advanced_Vector_Store_and_Search.ipynb`**: Added an embedding setup section and dynamically updated vector dimensions.

### 4. Testing Framework
- **New Test File**: `tests/test_all_features.py` - A comprehensive test suite covering 8 critical areas:
    1.  Embedding Generation
    2.  Provider Selection
    3.  **Dynamic Model Switching** (New)
    4.  Vector Store Operations
    5.  Metadata Filtering
    6.  Hybrid Search
    7.  Convenience Functions
    8.  Error Handling
- **Existing Tests**: Verified compatibility with existing tests (`tests/test_model_selection.py`, etc.).
- **CI/CD**: `pytest` passes all tests with Exit Code 0.

## How to Test
1.  Checkout the branch:
    ```bash
    git checkout embeddings
    ```
2.  Run the comprehensive test suite:
    ```bash
    python tests/test_all_features.py
    ```
3.  Run the full regression suite:
    ```bash
    pytest tests/
    ```

## Checklist
- [x] Code follows the style guidelines of this project
- [x] Documentation has been updated
- [x] Jupyter Notebooks (Cookbooks) have been tested and updated
- [x] New unit tests have been added
- [x] All existing tests pass
