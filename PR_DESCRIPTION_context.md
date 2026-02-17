# PR: Context Graph Decision Trace Hardening + Schema Compatibility

## Summary
This PR upgrades Semantica's context decision stack from partially-wired trace capture to a true end-to-end decision trace workflow in existing modules, while keeping backward compatibility.

It focuses on:
- Capturing decision traces with cross-system context, policy applications, exceptions, approvals, and precedents.
- Adding immutable-style hash-chained trace events (`DecisionTraceEvent`) for audit lineage.
- Hardening query/deserialization behavior across backend result shapes.
- Fixing schema/versioning mismatches and a few reliability bugs in context APIs.

## Why This Change
The context module already had strong building blocks (decision recorder, policy engine, causal analysis, retriever), but one key entrypoint was still a placeholder and some schema/query edges were inconsistent for production context-graph usage.

This PR closes those gaps so context graphs can be used as a decision system-of-record layer with better replayability and explainability.

## Scope
Changed files:
- `semantica/context/decision_methods.py`
- `semantica/context/graph_schema.py`
- `semantica/context/decision_query.py`
- `semantica/context/causal_analyzer.py`
- `semantica/context/context_retriever.py`
- `semantica/context/agent_context.py`
- `tests/context/test_decision_methods_trace.py`

No new modules were introduced. All changes are within existing context components.

## Detailed Changes

### 1. semantica/context/decision_methods.py ✨ Enhancement +217/-5

Implement end-to-end decision trace capture

• Replaced placeholder `capture_decision_trace()` with full orchestration logic  
• Added support for recording decisions, policies, exceptions, approvals, and precedents  
• Implemented `_append_immutable_trace_events()` for hash-chained audit trail generation  
• Made `graph_store` parameter optional for backward compatibility  
• Added `immutable_audit_log` flag to control trace event creation  
`semantica/context/decision_methods.py`

### 1) End-to-end trace capture (replaced placeholder)
File:
- `semantica/context/decision_methods.py`

Changes:
- Replaced placeholder `capture_decision_trace(...)` with actual orchestration:
  - Records decision.
  - Captures cross-system context.
  - Applies policy links.
  - Records exceptions.
  - Records approval chains.
  - Links precedents.
- Added optional immutable audit trail generation via hash-chained trace events.
- Added helper `_append_immutable_trace_events(...)`.

New signature:
```python
capture_decision_trace(
    decision,
    cross_system_context,
    graph_store=None,
    entities=None,
    source_documents=None,
    policy_ids=None,
    exceptions=None,
    approvals=None,
    precedents=None,
    immutable_audit_log=True
)
```

Behavior:
- If `graph_store` is omitted, behavior remains backward compatible and returns `decision_id` without failing.
- If `graph_store` is provided, full trace persistence is executed.

### 2) Immutable trace event schema + policy versioning compatibility
File:
- `semantica/context/graph_schema.py`

Changes:
- Added `DecisionTraceEvent` support:
  - Constraint: `decision_trace_id_unique`
  - Indexes: `decision_trace_id_index`, `decision_trace_event_index`, `decision_trace_type_index`, `decision_trace_timestamp_index`
  - Included in schema metadata and cleanup routines.
- Added relationship family metadata for:
  - `HAS_TRACE_EVENT`
  - `NEXT_TRACE_EVENT`
- Fixed policy schema mismatch:
  - Introduced composite identity constraint for policies:
    - `policy_identity_unique` on `(policy_id, version)`
  - Added fallback to legacy `policy_id_unique` when composite constraints are not supported by backend.

### 3) Decision query hardening for backend result shapes
File:
- `semantica/context/decision_query.py`

Changes:
- Made result parsing robust when records are returned as:
  - wrapped map (`{"d": {...}}`) or
  - flat map (`{"decision_id": ...}`).
- Improved ID handling:
  - accepts `decision_id` or `id`.
  - raises clear `KeyError` when required IDs are missing.
- Removed invalid dataclass constructor arg (`auto_generate_id=False`) from `Decision` and `PolicyException` construction.

### 4) Causal analyzer hardening for backend result shapes
File:
- `semantica/context/causal_analyzer.py`

Changes:
- Same robust record parsing strategy as decision query for `end`/`root` records.
- Improved ID handling (`decision_id` or `id`).
- Removed invalid dataclass constructor arg (`auto_generate_id=False`) from `Decision` construction.

### 5) Policy retrieval return-path fix
File:
- `semantica/context/context_retriever.py`

Changes:
- `_find_relevant_policies(...)` now always returns a list (previous success path could fall through without return).

### 6) AgentContext precedent conversion fix
File:
- `semantica/context/agent_context.py`

Changes:
- Fixed conversion from ContextGraph precedent dict to `Decision` object:
  - Removed unsupported `entities=` argument passed to `Decision`.
  - Moved entities into `metadata["entities"]` to preserve data and compatibility.

### 7) Added regression tests for trace capture
File:
- `tests/context/test_decision_methods_trace.py`

Tests added:
- Backward compatibility when `graph_store` is not passed.
- End-to-end trace path execution with `graph_store` mock and immutable audit logging enabled.

## Feature Alignment (Context Graph Requirements)
- ✅ Decision trace capture in execution path.
- ✅ Cross-system context persistence.
- ✅ Exception + approval + policy + precedent linkage in trace flow.
- ✅ Explainable lineage events with hash chaining.
- ✅ Temporal/event indexing for trace replay queries.
- ✅ Backward compatibility for existing convenience API usage.
- ⚠️ Storage-level tamper-proof guarantees are still backend-dependent (this PR adds app-level hash-chaining, not WORM storage).

## Backward Compatibility
- Existing call style still works:
  - `capture_decision_trace(decision, cross_system_context)` remains valid.
- No removed public APIs.
- Behavior is additive unless callers explicitly opt into richer trace fields.
- Policy constraint logic includes fallback for backends without composite-constraint support.

## Validation Performed
Compilation:
- `python -m compileall semantica/context/decision_methods.py semantica/context/graph_schema.py semantica/context/context_retriever.py semantica/context/agent_context.py semantica/context/decision_query.py semantica/context/causal_analyzer.py`

Targeted tests run and passed:
- `pytest -q tests/context/test_decision_methods_trace.py`
- `pytest -q tests/context/test_context_retriever_hybrid.py -k "find_relevant_policies"`
- `pytest -q tests/context/test_agent_context_decisions.py -k "find_precedents_success or record_decision_success"`
- `pytest -q tests/context/test_decision_query.py -k "find_precedents_hybrid_success"`
- `pytest -q tests/context/test_causal_analyzer.py -k "get_causal_chain_upstream_success"`

Note:
- Full repository test suite was not run in this PR scope.

## Risks and Mitigations
- Risk: Backend variations in Cypher support.
  - Mitigation: policy constraint fallback and tolerant query parsing.
- Risk: Extra trace writes increase write volume.
  - Mitigation: trace event creation is optional via `immutable_audit_log`.
- Risk: Existing consumers expecting entities directly on `Decision` constructor.
  - Mitigation: entities preserved in metadata; no constructor change required.

## Rollout Notes
- No migration is strictly required for existing users.
- To enable full trace recording, callers should pass `graph_store` to `capture_decision_trace(...)`.
- Existing deployments can adopt immutable tracing incrementally.

## Checklist
- [x] Implemented end-to-end decision trace capture.
- [x] Added immutable hash-chained trace events.
- [x] Updated schema with trace labels/indexes/constraints.
- [x] Fixed policy versioning constraint compatibility.
- [x] Hardened query/parsing paths for heterogeneous backend results.
- [x] Preserved backward compatibility for existing convenience calls.
- [x] Added targeted regression tests.
