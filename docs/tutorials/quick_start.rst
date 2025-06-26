Quick Start Tutorial
===================

This tutorial will guide you through your first steps with SemantiCore, from installation to processing your first document.

Prerequisites
-------------

- Python 3.8 or higher
- pip package manager
- Basic knowledge of Python

Installation
------------

**Step 1: Install SemantiCore**

.. code-block:: bash

   # Complete installation with all features
   pip install "semanticore[all]"

   # Or lightweight installation
   pip install semanticore

**Step 2: Verify Installation**

.. code-block:: python

   import semanticore
   print(f"SemantiCore version: {semanticore.__version__}")

Your First Document
-------------------

**Step 1: Create a Sample Document**

Create a file named `sample.txt` with the following content:

.. code-block:: text

   Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976.
   The company is headquartered in Cupertino, California.
   Tim Cook became CEO in 2011 after Steve Jobs passed away.
   Apple's revenue in 2023 was $394.33 billion.

**Step 2: Process the Document**

.. code-block:: python

   from semanticore import SemantiCore

   # Initialize SemantiCore
   core = SemantiCore()

   # Process the document
   result = core.process_document("sample.txt")

   # View results
   print(f"Entities found: {len(result.entities)}")
   print(f"Triples generated: {len(result.triples)}")
   print(f"Embeddings created: {len(result.embeddings)}")

**Step 3: Explore the Results**

.. code-block:: python

   # View extracted entities
   for entity in result.entities:
       print(f"Entity: {entity.text} (Type: {entity.type})")

   # View generated triples
   for triple in result.triples:
       print(f"Triple: {triple.subject} | {triple.predicate} | {triple.object}")

   # View embeddings
   print(f"Embedding dimension: {len(result.embeddings[0])}")

Working with Different Formats
------------------------------

**PDF Documents**

.. code-block:: python

   from semanticore.processors import DocumentProcessor

   # Initialize PDF processor
   pdf_processor = DocumentProcessor()

   # Process PDF
   pdf_result = pdf_processor.process_pdf("document.pdf")
   print(f"Extracted text: {len(pdf_result.text)} characters")

**Web Content**

.. code-block:: python

   from semanticore.processors import WebProcessor

   # Initialize web processor
   web_processor = WebProcessor()

   # Process web page
   web_result = web_processor.process_url("https://example.com/article")
   print(f"Title: {web_result.title}")
   print(f"Content: {len(web_result.content)} characters")

**Structured Data**

.. code-block:: python

   from semanticore.processors import StructuredDataProcessor

   # Initialize structured data processor
   structured_processor = StructuredDataProcessor()

   # Process JSON data
   json_result = structured_processor.process_json("data.json")
   print(f"Schema: {json_result.schema}")

Building a Knowledge Graph
--------------------------

**Step 1: Initialize Knowledge Graph**

.. code-block:: python

   from semanticore.knowledge_graph import KnowledgeGraphBuilder

   # Initialize knowledge graph builder
   kg_builder = KnowledgeGraphBuilder(
       storage_backend="memory"  # Use in-memory storage for this tutorial
   )

**Step 2: Add Triples**

.. code-block:: python

   # Add triples from processed document
   for triple in result.triples:
       kg_builder.add_triple(
           subject=triple.subject,
           predicate=triple.predicate,
           object=triple.object
       )

**Step 3: Build and Query**

.. code-block:: python

   # Build the knowledge graph
   kg_builder.build()

   # Query the knowledge graph
   query_results = kg_builder.query("""
       MATCH (e:Entity {name: "Apple Inc."})
       RETURN e
   """)

   print(f"Query results: {len(query_results)}")

Creating Vector Embeddings
--------------------------

**Step 1: Initialize Embedder**

.. code-block:: python

   from semanticore.embeddings import SemanticEmbedder

   # Initialize semantic embedder
   embedder = SemanticEmbedder(
       model="text-embedding-3-large",
       dimension=1536
   )

**Step 2: Generate Embeddings**

.. code-block:: python

   # Generate embeddings for document chunks
   chunks = result.chunks
   embeddings = embedder.generate_embeddings(chunks)

   print(f"Generated {len(embeddings)} embeddings")
   print(f"Each embedding has {len(embeddings[0])} dimensions")

**Step 3: Semantic Search**

.. code-block:: python

   # Perform semantic search
   query = "Who founded Apple?"
   search_results = embedder.semantic_search(query, embeddings, top_k=3)

   for i, (chunk, score) in enumerate(search_results):
       print(f"{i+1}. Score: {score:.3f}")
       print(f"   Content: {chunk[:100]}...")

Real-Time Processing
--------------------

**Step 1: Set up Feed Processing**

.. code-block:: python

   from semanticore.streaming import FeedProcessor
   import asyncio

   async def process_feeds():
       # Initialize feed processor
       feed_processor = FeedProcessor(
           update_interval="5m",
           deduplicate=True
       )

       # Subscribe to a feed
       feed_processor.subscribe("https://feeds.feedburner.com/TechCrunch")

       # Process items
       async for item in feed_processor.stream():
           print(f"New item: {item.title}")
           
           # Process with SemantiCore
           item_result = core.process_document(item.content)
           print(f"Extracted {len(item_result.entities)} entities")

   # Run the feed processor
   asyncio.run(process_feeds())

Configuration
-------------

**Environment Variables**

.. code-block:: bash

   export SEMANTICORE_LLM_PROVIDER=openai
   export SEMANTICORE_EMBEDDING_MODEL=text-embedding-3-large
   export OPENAI_API_KEY=your_api_key_here

**Configuration File**

Create `config.yaml`:

.. code-block:: yaml

   llm:
     provider: openai
     model: gpt-4
     api_key: ${OPENAI_API_KEY}

   embeddings:
     model: text-embedding-3-large
     dimension: 1536

   processing:
     batch_size: 100
     max_workers: 4

**Programmatic Configuration**

.. code-block:: python

   config = {
       "llm": {
           "provider": "openai",
           "model": "gpt-4",
           "api_key": "your_api_key"
       },
       "embeddings": {
           "model": "text-embedding-3-large",
           "dimension": 1536
       }
   }

   core = SemantiCore(config=config)

Error Handling
--------------

**Basic Error Handling**

.. code-block:: python

   from semanticore.core.exceptions import SemantiCoreError

   try:
       result = core.process_document("document.pdf")
   except SemantiCoreError as e:
       print(f"Error processing document: {e}")
   except FileNotFoundError:
       print("Document not found")
   except Exception as e:
       print(f"Unexpected error: {e}")

**Validation**

.. code-block:: python

   # Validate input before processing
   import os

   def process_safe(file_path):
       if not os.path.exists(file_path):
           raise FileNotFoundError(f"File not found: {file_path}")
       
       if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
           raise ValueError("File too large")
       
       return core.process_document(file_path)

Performance Optimization
------------------------

**Batch Processing**

.. code-block:: python

   # Process multiple documents efficiently
   documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
   
   results = core.process_documents(documents)
   
   for doc, result in zip(documents, results):
       print(f"{doc}: {len(result.entities)} entities")

**Memory Management**

.. code-block:: python

   # Process large datasets with generators
   def document_generator():
       for doc in large_dataset:
           yield doc

   for result in core.process_documents_stream(document_generator()):
       process_result(result)

Next Steps
----------

Congratulations! You've completed the quick start tutorial. Here's what you can explore next:

1. **Advanced Examples**: Check out the comprehensive examples in the examples section
2. **API Reference**: Explore the complete API documentation
3. **Tutorials**: Follow step-by-step tutorials for specific use cases
4. **Community**: Join our Discord community for help and discussions

**Additional Resources**

- 📖 [Complete Documentation](https://semanticore.readthedocs.io/)
- 💡 [Examples Repository](https://github.com/semanticore/examples)
- 💬 [Community Discord](https://discord.gg/semanticore)
- 🐙 [GitHub Repository](https://github.com/semanticore/semanticore)

.. raw:: html

   <div style="text-align: center; margin: 20px 0; padding: 15px; background-color: #d4edda; border-left: 4px solid #27AE60; border-radius: 5px;">
       <strong>🎉 Congratulations!</strong> You've successfully completed the SemantiCore quick start tutorial. You're now ready to transform your data into intelligent knowledge!
   </div> 