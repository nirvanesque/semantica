Getting Started with SemantiCore
================================

Welcome to SemantiCore! This guide will help you get up and running with the most comprehensive semantic data transformation toolkit.

Installation
------------

Choose the installation option that best fits your needs:

**Complete Installation (Recommended)**
.. code-block:: bash

   pip install "semanticore[all]"

**Lightweight Installation**
.. code-block:: bash

   pip install semanticore

**Specific Format Support**
.. code-block:: bash

   # PDF and Office documents
   pip install "semanticore[pdf,office]"

   # Web content and feeds
   pip install "semanticore[web,feeds]"

   # Database and vector stores
   pip install "semanticore[database,vector]"

   # Machine learning capabilities
   pip install "semanticore[ml]"

**Development Installation**
.. code-block:: bash

   git clone https://github.com/semanticore/semanticore.git
   cd semanticore
   pip install -e ".[dev]"

Quick Start
-----------

**30-Second Demo: From Any Format to Knowledge**

.. code-block:: python

   from semanticore import SemantiCore

   # Initialize with preferred providers
   core = SemantiCore(
       llm_provider="openai",
       embedding_model="text-embedding-3-large",
       vector_store="pinecone",
       graph_db="neo4j"
   )

   # Process ANY data format
   sources = [
       "financial_report.pdf",
       "https://example.com/news/rss",
       "research_papers/",
       "data.json",
       "https://example.com/article"
   ]

   # One-line semantic transformation
   knowledge_base = core.build_knowledge_base(sources)

   print(f"Processed {len(knowledge_base.documents)} documents")
   print(f"Extracted {len(knowledge_base.entities)} entities")
   print(f"Generated {len(knowledge_base.triples)} semantic triples")
   print(f"Created {len(knowledge_base.embeddings)} vector embeddings")

   # Query the knowledge base
   results = knowledge_base.query("What are the key financial trends?")

Basic Usage Examples
--------------------

**1. Document Processing**

.. code-block:: python

   from semanticore.processors import DocumentProcessor

   # Initialize document processor
   doc_processor = DocumentProcessor(
       extract_tables=True,
       extract_images=True,
       extract_metadata=True,
       preserve_structure=True
   )

   # Process various document types
   pdf_content = doc_processor.process_pdf("report.pdf")
   docx_content = doc_processor.process_docx("document.docx")
   pptx_content = doc_processor.process_pptx("presentation.pptx")

**2. Web Content Processing**

.. code-block:: python

   from semanticore.processors import WebProcessor, FeedProcessor

   # Web content processor
   web_processor = WebProcessor(
       respect_robots=True,
       extract_metadata=True,
       follow_redirects=True,
       max_depth=3
   )

   # RSS/Atom feed processor
   feed_processor = FeedProcessor(
       update_interval="5m",
       deduplicate=True,
       extract_full_content=True
   )

   # Process web content
   webpage = web_processor.process_url("https://example.com/article")
   semantics = core.extract_semantics(webpage.content)

**3. Semantic Extraction**

.. code-block:: python

   from semanticore.extraction import TripleExtractor

   # Initialize triple extractor
   triple_extractor = TripleExtractor(
       confidence_threshold=0.8,
       include_implicit_relations=True,
       temporal_modeling=True
   )

   # Extract triples from any content
   text = "Apple Inc. was founded by Steve Jobs in 1976 in Cupertino, California."
   triples = triple_extractor.extract_triples(text)

   print(triples)
   # [
   #   Triple(subject="Apple Inc.", predicate="founded_by", object="Steve Jobs"),
   #   Triple(subject="Apple Inc.", predicate="founded_in", object="1976"),
   #   Triple(subject="Apple Inc.", predicate="located_in", object="Cupertino"),
   #   Triple(subject="Cupertino", predicate="located_in", object="California")
   # ]

**4. Knowledge Graph Construction**

.. code-block:: python

   from semanticore.knowledge_graph import KnowledgeGraphBuilder

   # Initialize knowledge graph builder
   kg_builder = KnowledgeGraphBuilder(
       storage_backend="neo4j",
       uri="bolt://localhost:7687",
       username="neo4j",
       password="password"
   )

   # Build knowledge graph from triples
   kg_builder.add_triples(triples)
   kg_builder.build()

   # Query the knowledge graph
   results = kg_builder.query("""
       MATCH (e:Entity {name: "Apple Inc."})
       RETURN e
   """)

**5. Vector Embeddings**

.. code-block:: python

   from semanticore.embeddings import SemanticEmbedder

   # Initialize semantic embedder
   embedder = SemanticEmbedder(
       model="text-embedding-3-large",
       dimension=1536,
       preserve_context=True,
       semantic_chunking=True
   )

   # Generate semantic embeddings
   documents = load_documents()
   semantic_chunks = embedder.semantic_chunk(documents)
   embeddings = embedder.generate_embeddings(semantic_chunks)

   # Store in vector database
   vector_store = core.get_vector_store("pinecone")
   vector_store.store_embeddings(semantic_chunks, embeddings)

   # Semantic search
   query = "artificial intelligence applications in healthcare"
   results = vector_store.semantic_search(query, top_k=10)

Configuration
-------------

SemantiCore can be configured through environment variables or a configuration file:

**Environment Variables**
.. code-block:: bash

   export SEMANTICORE_LLM_PROVIDER=openai
   export SEMANTICORE_EMBEDDING_MODEL=text-embedding-3-large
   export SEMANTICORE_VECTOR_STORE=pinecone
   export SEMANTICORE_GRAPH_DB=neo4j
   export OPENAI_API_KEY=your_openai_api_key
   export PINECONE_API_KEY=your_pinecone_api_key

**Configuration File (config.yaml)**
.. code-block:: yaml

   llm:
     provider: openai
     model: gpt-4
     api_key: ${OPENAI_API_KEY}

   embeddings:
     model: text-embedding-3-large
     dimension: 1536
     preserve_context: true

   vector_store:
     provider: pinecone
     api_key: ${PINECONE_API_KEY}
     environment: us-west1-gcp

   knowledge_graph:
     provider: neo4j
     uri: bolt://localhost:7687
     username: neo4j
     password: password

   processing:
     batch_size: 100
     max_workers: 4
     timeout: 300

Next Steps
----------

Now that you have SemantiCore installed and running, explore these resources:

* :doc:`examples` - Comprehensive examples for different use cases
* :doc:`tutorials/index` - Step-by-step tutorials
* :doc:`api/index` - Complete API reference
* :doc:`advanced/streaming` - Real-time processing capabilities
* :doc:`advanced/deployment` - Production deployment guide

Common Issues
-------------

**Import Error: No module named 'semanticore'**
   Make sure you've installed SemantiCore correctly. Try reinstalling with:
   .. code-block:: bash
      pip install --upgrade semanticore

**API Key Errors**
   Ensure your API keys are set correctly in environment variables or configuration file.

**Memory Issues**
   For large documents, consider increasing your system's memory or using batch processing.

**Performance Issues**
   Enable GPU acceleration if available and adjust batch sizes in configuration.

Need Help?
----------

* 📖 **Documentation**: Browse the complete documentation
* 💬 **Discord**: Join our community for real-time support
* 🐙 **GitHub**: Report issues and contribute
* 📧 **Email**: Contact us at support@semanticore.io

.. raw:: html

   <div style="text-align: center; margin: 20px 0; padding: 15px; background-color: #e8f4fd; border-left: 4px solid #2980B9; border-radius: 5px;">
       <strong>💡 Pro Tip:</strong> Start with the lightweight installation and add specific format support as needed. This keeps your environment clean and reduces dependencies. 