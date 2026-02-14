# Context: PolicyEngine fixes, new context tests, cleanup — all tests passing

## Summary
This PR improves Context Graph decision tracking reliability by fixing PolicyEngine behavior on non‑Cypher backends, resolving a compile issue, and adding focused tests. It also removes an example script per request. All tests pass.

- Source branch: `context`
- Base branch: do not push directly to `main`; open PR for review
- Scope: Context graphs decision tracking and policy evaluation

## Changes

### Policy engine
- Fix: Correct indentation for Cypher branches to prevent compile errors.
- Fix: `get_policy` (non‑Cypher path) now selects the latest policy by semantic version first, falling back to `updated_at` as a secondary key.

### Tests
- Added `tests/context/test_policy_engine_fallback.py` for ContextGraph (no Cypher) versioning, compliance, and `APPLIED_POLICY` edges.
- Added `tests/context/test_agent_context_smoke.py` for AgentContext + ContextGraph decision tracking (record decisions, causal chain, policy storage).

### Cleanup
- Removed `examples/context_graphs_decision_tracking.py`.
- Deleted transient test artifacts from earlier local runs (not tracked).

## Files Touched (high level)
- `semantica/context/policy_engine.py`
- `tests/context/test_policy_engine_fallback.py` (new)
- `tests/context/test_agent_context_smoke.py` (new)
- `examples/context_graphs_decision_tracking.py` (removed)

## Rationale
- PolicyEngine compile error: Cypher code path indentation was causing a compile failure; fixed so it executes only on Cypher‑capable backends.
- Latest version retrieval: On the in‑memory ContextGraph backend, latest policy selection relied on `updated_at` strings and could return the wrong version. Updated to prefer semantic version order with robust fallback.

## Test Coverage

### New tests
- PolicyEngine fallback on ContextGraph:
  - Adds/gets/updates policies and verifies the latest version is returned post‑update.
  - Checks compliance and presence of `APPLIED_POLICY` edges.
- AgentContext + ContextGraph smoke:
  - Records decisions, links causal relationship, validates upstream chain retrieval.
  - Stores and retrieves policies via PolicyEngine obtained from AgentContext.

### Existing tests (already in repo) cover
- ContextGraph decision nodes, precedents, and edge cases.
- Banking/Healthcare end‑to‑end flows.
- Context retriever hybrid/precedents and AgentContext integration.

## Results
- Context-only tests:
  - `python -m pytest -q tests/context` → passed
- Full test suite:
  - `python -m pytest -q tests` → passed
- Example subset (smoke):
  - `tests/context/test_agent_context_smoke.py` → passed (deprecation warnings are expected)
  - `tests/context/test_policy_engine_fallback.py` → passed after fixes

## How to Test Locally

```bash
# Optional: fresh venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dev deps (includes pytest)
pip install -e ".[dev]"

# Run context suite
python -m pytest -q tests\context

# Run full suite
python -m pytest -q tests
```

## Backwards Compatibility & Risk
- No breaking API changes; fixes are internal behavior corrections for non‑Cypher PolicyEngine operations.
- Risk is low; covered by new tests and full suite.

## Follow‑ups (optional)
- Add stricter warnings policy if desired (e.g., `filterwarnings` in `pyproject.toml`/`pytest.ini`).
- Expand negative tests (e.g., cycles in causal traversal, invalid rule sets).

## Checklist
- [x] No direct pushes to `main`.
- [x] New tests added and passing locally.
- [x] No secrets or credentials added.
- [x] Example file removed per request.
- [x] Clear rationale and instructions provided.

