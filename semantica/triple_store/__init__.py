"""
Triple Store Module

This module provides comprehensive triple store integration and management
for RDF data storage and querying, supporting multiple triple store backends
with unified interfaces.

Key Features:
    - Multi-backend support (Blazegraph, Jena, RDF4J, Virtuoso)
    - CRUD operations for RDF triples
    - SPARQL query execution and optimization
    - Bulk data loading with progress tracking
    - Query caching and optimization
    - Transaction support
    - Store adapter pattern

Main Classes:
    - TripleManager: Main triple store management coordinator
    - QueryEngine: SPARQL query execution and optimization
    - BulkLoader: High-volume data loading
    - BlazegraphAdapter: Blazegraph integration adapter
    - JenaAdapter: Apache Jena integration adapter
    - RDF4JAdapter: Eclipse RDF4J integration adapter
    - VirtuosoAdapter: Virtuoso RDF store integration adapter
    - TripleStore: Triple store configuration dataclass
    - QueryResult: Query result representation dataclass
    - QueryPlan: Query execution plan dataclass
    - LoadProgress: Bulk loading progress dataclass

Example Usage:
    >>> from semantica.triple_store import TripleManager
    >>> manager = TripleManager()
    >>> store = manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph")
    >>> result = manager.add_triple(triple, store_id="main")
    >>> from semantica.triple_store import QueryEngine
    >>> engine = QueryEngine()
    >>> query_result = engine.execute_query(sparql_query, store_adapter)

Author: Semantica Contributors
License: MIT
"""

from .triple_manager import (
    TripleManager,
    TripleStore
)
from .blazegraph_adapter import (
    BlazegraphAdapter
)
from .jena_adapter import (
    JenaAdapter
)
from .rdf4j_adapter import (
    RDF4JAdapter
)
from .virtuoso_adapter import (
    VirtuosoAdapter
)
from .query_engine import (
    QueryEngine,
    QueryResult,
    QueryPlan
)
from .bulk_loader import (
    BulkLoader,
    LoadProgress
)

__all__ = [
    # Triple management
    "TripleManager",
    "TripleStore",
    
    # Store adapters
    "BlazegraphAdapter",
    "JenaAdapter",
    "RDF4JAdapter",
    "VirtuosoAdapter",
    
    # Query engine
    "QueryEngine",
    "QueryResult",
    "QueryPlan",
    
    # Bulk loading
    "BulkLoader",
    "LoadProgress",
]
