API Reference
=============

Welcome to the SemantiCore API reference. This section provides comprehensive documentation for all SemantiCore modules, classes, and functions.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   core
   processors
   extraction
   embeddings
   knowledge_graph
   streaming
   domains

Quick API Overview
------------------

**Main Entry Point**
.. code-block:: python

   from semanticore import SemantiCore

   # Initialize the main engine
   core = SemantiCore(
       llm_provider="openai",
       embedding_model="text-embedding-3-large",
       vector_store="pinecone",
       graph_db="neo4j"
   )

**Document Processing**
.. code-block:: python

   from semanticore.processors import DocumentProcessor

   processor = DocumentProcessor()
   result = processor.process("document.pdf")

**Semantic Extraction**
.. code-block:: python

   from semanticore.extraction import TripleExtractor

   extractor = TripleExtractor()
   triples = extractor.extract_triples(text)

**Knowledge Graph**
.. code-block:: python

   from semanticore.knowledge_graph import KnowledgeGraphBuilder

   builder = KnowledgeGraphBuilder()
   builder.add_triples(triples)
   builder.build()

**Vector Embeddings**
.. code-block:: python

   from semanticore.embeddings import SemanticEmbedder

   embedder = SemanticEmbedder()
   embeddings = embedder.generate_embeddings(documents)

Module Structure
----------------

.. code-block:: text

   semanticore/
   ├── core/                    # Core framework
   │   ├── engine.py           # Main SemantiCore engine
   │   ├── config.py           # Configuration management
   │   └── exceptions.py       # Custom exceptions
   │
   ├── processors/             # Data processing modules
   │   ├── document/           # Document processing
   │   ├── web/               # Web content processing
   │   ├── structured/        # Structured data processing
   │   └── base.py            # Base processor class
   │
   ├── extraction/            # Semantic extraction
   │   ├── entities.py        # Entity extraction
   │   ├── relationships.py   # Relationship extraction
   │   └── triples.py         # Triple generation
   │
   ├── embeddings/            # Embedding generation
   │   ├── text_embeddings.py
   │   └── vector_stores.py
   │
   ├── knowledge_graph/       # Knowledge graph construction
   │   ├── builder.py
   │   └── storage.py
   │
   ├── streaming/             # Real-time processing
   │   └── feed_processor.py
   │
   └── domains/               # Domain-specific processors
       ├── cybersecurity/
       ├── biomedical/
       └── finance/

Configuration
-------------

SemantiCore can be configured through various methods:

**Environment Variables**
.. code-block:: bash

   export SEMANTICORE_LLM_PROVIDER=openai
   export SEMANTICORE_EMBEDDING_MODEL=text-embedding-3-large
   export SEMANTICORE_VECTOR_STORE=pinecone
   export SEMANTICORE_GRAPH_DB=neo4j

**Configuration File**
.. code-block:: yaml

   llm:
     provider: openai
     model: gpt-4
     api_key: ${OPENAI_API_KEY}

   embeddings:
     model: text-embedding-3-large
     dimension: 1536

   vector_store:
     provider: pinecone
     api_key: ${PINECONE_API_KEY}

   knowledge_graph:
     provider: neo4j
     uri: bolt://localhost:7687

**Programmatic Configuration**
.. code-block:: python

   config = {
       "llm": {
           "provider": "openai",
           "model": "gpt-4",
           "api_key": "your-api-key"
       },
       "embeddings": {
           "model": "text-embedding-3-large",
           "dimension": 1536
       }
   }

   core = SemantiCore(config=config)

Error Handling
--------------

SemantiCore provides comprehensive error handling:

.. code-block:: python

   from semanticore.core.exceptions import (
       SemantiCoreError,
       ProcessingError,
       ConfigurationError,
       ValidationError
   )

   try:
       result = core.process_document("document.pdf")
   except ProcessingError as e:
       print(f"Processing failed: {e}")
   except ConfigurationError as e:
       print(f"Configuration error: {e}")
   except SemantiCoreError as e:
       print(f"General error: {e}")

Type Hints
----------

All SemantiCore functions include comprehensive type hints:

.. code-block:: python

   from typing import List, Dict, Optional, Union
   from semanticore.core.types import (
       ProcessedContent,
       Entity,
       Triple,
       Embedding,
       KnowledgeBase
   )

   def process_documents(
       self,
       sources: List[str],
       config: Optional[Dict] = None
   ) -> List[ProcessedContent]:
       """Process multiple documents."""
       pass

Performance Considerations
-------------------------

**Batch Processing**
.. code-block:: python

   # Process documents in batches for better performance
   batch_size = 100
   for i in range(0, len(documents), batch_size):
       batch = documents[i:i + batch_size]
       results = core.process_documents(batch)

**Memory Management**
.. code-block:: python

   # Use generators for large datasets
   def document_generator():
       for doc in large_dataset:
           yield doc

   for result in core.process_documents_stream(document_generator()):
       process_result(result)

**Parallel Processing**
.. code-block:: python

   # Enable parallel processing
   core = SemantiCore(
       config={
           "processing": {
               "max_workers": 4,
               "batch_size": 50
           }
       }
   )

Best Practices
--------------

1. **Use appropriate batch sizes** for your hardware
2. **Handle errors gracefully** with try-catch blocks
3. **Monitor memory usage** for large datasets
4. **Use type hints** for better code quality
5. **Configure logging** for debugging
6. **Validate inputs** before processing
7. **Use async/await** for I/O operations
8. **Cache results** when appropriate

.. raw:: html

   <div style="text-align: center; margin: 20px 0; padding: 15px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 5px;">
       <strong>📚 Note:</strong> This API reference is generated from the source code. For the most up-to-date information, check the source code or run <code>help()</code> on any SemantiCore object.
   </div> 