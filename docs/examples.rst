SemantiCore Examples
====================

This page provides comprehensive examples of SemantiCore usage across different domains and use cases.

Basic Examples
--------------

**Simple Document Processing**

.. code-block:: python

   from semanticore import SemantiCore

   # Initialize SemantiCore
   core = SemantiCore()

   # Process a single document
   result = core.process_document("financial_report.pdf")
   
   print(f"Extracted {len(result.entities)} entities")
   print(f"Generated {len(result.triples)} triples")
   print(f"Created {len(result.embeddings)} embeddings")

**Batch Processing Multiple Documents**

.. code-block:: python

   from semanticore import SemantiCore

   core = SemantiCore()

   # Process multiple documents
   documents = [
       "report1.pdf",
       "report2.docx",
       "data.json",
       "https://example.com/article"
   ]

   results = core.process_documents(documents)
   
   for doc, result in zip(documents, results):
       print(f"{doc}: {len(result.entities)} entities, {len(result.triples)} triples")

**Custom Configuration**

.. code-block:: python

   from semanticore import SemantiCore

   # Initialize with custom configuration
   core = SemantiCore(
       llm_provider="openai",
       embedding_model="text-embedding-3-large",
       vector_store="pinecone",
       graph_db="neo4j",
       config={
           "processing": {
               "batch_size": 50,
               "max_workers": 4,
               "timeout": 300
           },
           "extraction": {
               "confidence_threshold": 0.8,
               "include_implicit_relations": True
           }
       }
   )

Document Processing Examples
---------------------------

**PDF Document with Tables and Images**

.. code-block:: python

   from semanticore.processors.document import PDFProcessor

   # Initialize PDF processor with advanced features
   pdf_processor = PDFProcessor(
       extract_tables=True,
       extract_images=True,
       extract_metadata=True,
       preserve_structure=True,
       ocr_enabled=True
   )

   # Process PDF
   result = pdf_processor.process("financial_report.pdf")
   
   # Access extracted content
   print(f"Text content: {len(result.text)} characters")
   print(f"Tables extracted: {len(result.tables)}")
   print(f"Images extracted: {len(result.images)}")
   print(f"Metadata: {result.metadata}")

**Office Documents (Word, PowerPoint, Excel)**

.. code-block:: python

   from semanticore.processors.document import (
       DOCXProcessor, PPTXProcessor, ExcelProcessor
   )

   # Process Word document
   docx_processor = DOCXProcessor(extract_comments=True)
   docx_result = docx_processor.process("document.docx")

   # Process PowerPoint presentation
   pptx_processor = PPTXProcessor(extract_notes=True)
   pptx_result = pptx_processor.process("presentation.pptx")

   # Process Excel spreadsheet
   excel_processor = ExcelProcessor(extract_formulas=True)
   excel_result = excel_processor.process("data.xlsx")

   # Combine results
   all_content = docx_result + pptx_result + excel_result

Web Content Processing
----------------------

**RSS Feed Monitoring**

.. code-block:: python

   from semanticore.processors.web import FeedProcessor
   import asyncio

   async def monitor_feeds():
       # Initialize feed processor
       feed_processor = FeedProcessor(
           update_interval="5m",
           deduplicate=True,
           extract_full_content=True
       )

       # Subscribe to feeds
       feeds = [
           "https://feeds.feedburner.com/TechCrunch",
           "https://rss.cnn.com/rss/edition.rss",
           "https://feeds.reuters.com/reuters/topNews"
       ]

       for feed_url in feeds:
           feed_processor.subscribe(feed_url, category="news")

       # Process items in real-time
       async for feed_item in feed_processor.stream():
           print(f"New item: {feed_item.title}")
           
           # Extract semantics
           semantics = core.extract_semantics(feed_item.content)
           triples = core.generate_triples(semantics)
           
           # Update knowledge graph
           knowledge_graph.add_triples(triples)

   # Run the feed monitor
   asyncio.run(monitor_feeds())

**Web Scraping with Semantic Understanding**

.. code-block:: python

   from semanticore.processors.web import WebProcessor

   # Initialize web processor
   web_processor = WebProcessor(
       respect_robots=True,
       extract_metadata=True,
       follow_redirects=True,
       max_depth=3,
       user_agent="SemantiCore Bot/1.0"
   )

   # Process web pages
   urls = [
       "https://example.com/article1",
       "https://example.com/article2",
       "https://example.com/article3"
   ]

   for url in urls:
       webpage = web_processor.process_url(url)
       
       # Extract semantic information
       semantics = core.extract_semantics(webpage.content)
       entities = core.extract_entities(webpage.content)
       triples = core.generate_triples(semantics)
       
       print(f"URL: {url}")
       print(f"Entities: {len(entities)}")
       print(f"Triples: {len(triples)}")

Semantic Extraction Examples
----------------------------

**Entity and Relationship Extraction**

.. code-block:: python

   from semanticore.extraction import EntityExtractor, RelationshipExtractor

   # Initialize extractors
   entity_extractor = EntityExtractor(
       model="en_core_web_sm",
       entity_types=["PERSON", "ORG", "GPE", "DATE", "MONEY"]
   )

   relationship_extractor = RelationshipExtractor(
       confidence_threshold=0.7,
       include_implicit_relations=True
   )

   # Extract entities and relationships
   text = """
   Apple Inc. was founded by Steve Jobs in 1976 in Cupertino, California.
   The company's revenue in 2023 was $394.33 billion.
   Tim Cook is the current CEO of Apple.
   """

   entities = entity_extractor.extract_entities(text)
   relationships = relationship_extractor.extract_relationships(text, entities)

   print("Entities found:")
   for entity in entities:
       print(f"  {entity.text} ({entity.type}) - Confidence: {entity.confidence}")

   print("\nRelationships found:")
   for rel in relationships:
       print(f"  {rel.subject} --{rel.predicate}--> {rel.object}")

**Triple Generation**

.. code-block:: python

   from semanticore.extraction import TripleExtractor

   # Initialize triple extractor
   triple_extractor = TripleExtractor(
       confidence_threshold=0.8,
       include_implicit_relations=True,
       temporal_modeling=True,
       spatial_modeling=True
   )

   # Extract triples from text
   text = """
   Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975.
   The company is headquartered in Redmond, Washington.
   Satya Nadella became CEO in 2014.
   Microsoft acquired LinkedIn in 2016 for $26.2 billion.
   """

   triples = triple_extractor.extract_triples(text)

   print("Generated Triples:")
   for triple in triples:
       print(f"  {triple.subject} | {triple.predicate} | {triple.object}")
       print(f"    Confidence: {triple.confidence:.2f}")

   # Export to different formats
   turtle_format = triple_extractor.to_turtle(triples)
   ntriples_format = triple_extractor.to_ntriples(triples)
   jsonld_format = triple_extractor.to_jsonld(triples)

Knowledge Graph Examples
------------------------

**Building a Knowledge Graph**

.. code-block:: python

   from semanticore.knowledge_graph import KnowledgeGraphBuilder

   # Initialize knowledge graph builder
   kg_builder = KnowledgeGraphBuilder(
       storage_backend="neo4j",
       uri="bolt://localhost:7687",
       username="neo4j",
       password="password"
   )

   # Add triples to the graph
   triples = [
       ("Apple Inc.", "founded_by", "Steve Jobs"),
       ("Apple Inc.", "founded_in", "1976"),
       ("Apple Inc.", "located_in", "Cupertino"),
       ("Cupertino", "located_in", "California"),
       ("Steve Jobs", "co_founded", "Apple Inc."),
       ("Tim Cook", "is_CEO_of", "Apple Inc."),
       ("Apple Inc.", "revenue_2023", "$394.33 billion")
   ]

   for subject, predicate, object in triples:
       kg_builder.add_triple(subject, predicate, object)

   # Build the knowledge graph
   kg_builder.build()

   # Query the knowledge graph
   results = kg_builder.query("""
       MATCH (e:Entity {name: "Apple Inc."})
       OPTIONAL MATCH (e)-[r]->(o)
       RETURN e.name, type(r), o.name
   """)

   for result in results:
       print(f"{result['e.name']} --{result['type(r)']}--> {result['o.name']}")

**SPARQL Query Generation**

.. code-block:: python

   from semanticore.knowledge_graph import SPARQLGenerator

   # Initialize SPARQL generator
   sparql_gen = SPARQLGenerator()

   # Generate SPARQL queries
   natural_query = "Who founded Apple Inc.?"
   sparql_query = sparql_gen.generate_sparql(natural_query)

   print(f"Natural Language: {natural_query}")
   print(f"SPARQL Query: {sparql_query}")

   # Execute the query
   results = kg_builder.execute_sparql(sparql_query)
   print(f"Results: {results}")

Vector Embeddings Examples
--------------------------

**Semantic Embeddings Generation**

.. code-block:: python

   from semanticore.embeddings import SemanticEmbedder

   # Initialize semantic embedder
   embedder = SemanticEmbedder(
       model="text-embedding-3-large",
       dimension=1536,
       preserve_context=True,
       semantic_chunking=True
   )

   # Generate embeddings for documents
   documents = [
       "Artificial intelligence is transforming healthcare.",
       "Machine learning algorithms improve patient outcomes.",
       "AI-powered diagnostics reduce medical errors.",
       "Healthcare technology advances rapidly."
   ]

   # Generate embeddings
   embeddings = embedder.generate_embeddings(documents)

   print(f"Generated {len(embeddings)} embeddings")
   print(f"Embedding dimension: {len(embeddings[0])}")

**Semantic Search**

.. code-block:: python

   from semanticore.embeddings import VectorStore

   # Initialize vector store
   vector_store = VectorStore(
       provider="pinecone",
       api_key="your_pinecone_api_key",
       environment="us-west1-gcp"
   )

   # Store embeddings
   vector_store.store_embeddings(documents, embeddings)

   # Semantic search
   query = "How is AI used in medical diagnosis?"
   results = vector_store.semantic_search(
       query=query,
       top_k=5,
       include_metadata=True
   )

   print(f"Search results for: {query}")
   for i, result in enumerate(results, 1):
       print(f"{i}. {result.document} (Score: {result.score:.3f})")

**Multi-Modal Embeddings**

.. code-block:: python

   from semanticore.embeddings import MultiModalEmbedder

   # Initialize multi-modal embedder
   mm_embedder = MultiModalEmbedder(
       text_model="text-embedding-3-large",
       image_model="clip-vit-base-patch32",
       audio_model="whisper-base"
   )

   # Generate multi-modal embeddings
   content = {
       "text": "A chart showing quarterly revenue growth",
       "image": "revenue_chart.png",
       "audio": "earnings_call.mp3"
   }

   embeddings = mm_embedder.generate_embeddings(content)
   
   print(f"Text embedding: {len(embeddings['text'])} dimensions")
   print(f"Image embedding: {len(embeddings['image'])} dimensions")
   print(f"Audio embedding: {len(embeddings['audio'])} dimensions")

Real-Time Processing Examples
-----------------------------

**Stream Processing with Kafka**

.. code-block:: python

   from semanticore.streaming import KafkaProcessor
   import asyncio

   async def process_stream():
       # Initialize Kafka processor
       kafka_processor = KafkaProcessor(
           bootstrap_servers=["localhost:9092"],
           topics=["documents", "web_content", "feeds"],
           group_id="semanticore-processor"
       )

       # Process streaming data
       async for message in kafka_processor.consume():
           content = message.value
           
           # Determine content type and process accordingly
           if message.headers.get("content_type") == "application/pdf":
               processed = doc_processor.process_pdf_bytes(content)
           elif message.headers.get("content_type") == "text/html":
               processed = web_processor.process_html(content)
           else:
               processed = content
           
           # Extract semantics and build knowledge
           semantics = core.extract_semantics(processed)
           triples = core.generate_triples(semantics)
           knowledge_graph.add_triples(triples)
           
           print(f"Processed message from topic: {message.topic}")

   # Run the stream processor
   asyncio.run(process_stream())

Domain-Specific Examples
------------------------

**Cybersecurity Intelligence**

.. code-block:: python

   from semanticore.domains.cybersecurity import CyberIntelProcessor

   # Initialize cybersecurity processor
   cyber_processor = CyberIntelProcessor(
       threat_feeds=[
           "https://feeds.feedburner.com/CyberSecurityNewsDaily",
           "https://www.us-cert.gov/ncas/current-activity.xml"
       ],
       formats=["pdf", "html", "xml", "json"],
       extract_iocs=True,
       map_to_mitre=True
   )

   # Process cybersecurity sources
   sources = [
       "threat_report.pdf",
       "https://security-blog.com/rss",
       "vulnerability_data.json"
   ]

   cyber_knowledge = cyber_processor.build_threat_intelligence(sources)

   # Generate STIX bundles
   stix_bundle = cyber_knowledge.to_stix()
   print(f"Generated STIX bundle with {len(stix_bundle.objects)} objects")

**Biomedical Literature Processing**

.. code-block:: python

   from semanticore.domains.biomedical import BiomedicalProcessor

   # Initialize biomedical processor
   bio_processor = BiomedicalProcessor(
       pubmed_integration=True,
       extract_drug_interactions=True,
       map_to_mesh=True,
       clinical_trial_detection=True
   )

   # Process biomedical literature
   sources = [
       "research_papers/",
       "https://pubmed.ncbi.nlm.nih.gov/rss/",
       "clinical_reports.pdf"
   ]

   biomedical_knowledge = bio_processor.build_medical_knowledge_base(sources)

   # Generate medical ontology
   medical_ontology = biomedical_knowledge.generate_ontology()

**Financial Data Analysis**

.. code-block:: python

   from semanticore.domains.finance import FinancialProcessor

   # Initialize financial processor
   finance_processor = FinancialProcessor(
       sec_filings=True,
       news_sentiment=True,
       market_data_integration=True,
       regulatory_compliance=True
   )

   # Process financial data sources
   sources = [
       "earnings_reports/",
       "https://feeds.finance.yahoo.com/rss/",
       "sec_filings.xml"
   ]

   financial_knowledge = finance_processor.build_financial_knowledge_graph(sources)

   # Generate financial semantic triples
   triples = financial_knowledge.extract_financial_triples()

Integration Examples
--------------------

**LangChain Integration**

.. code-block:: python

   from langchain.llms import OpenAI
   from langchain.chains import LLMChain
   from semanticore.integrations.langchain import SemantiCoreLoader

   # Initialize LangChain components
   llm = OpenAI(temperature=0)
   chain = LLMChain(llm=llm, prompt=prompt)

   # Use SemantiCore as a document loader
   loader = SemantiCoreLoader(
       sources=["documents/"],
       extract_semantics=True,
       generate_triples=True
   )

   documents = loader.load()

   # Process with LangChain
   for doc in documents:
       result = chain.run(doc.page_content)
       print(f"Analysis: {result}")

**Streamlit Web Application**

.. code-block:: python

   import streamlit as st
   from semanticore import SemantiCore

   st.title("SemantiCore Document Processor")

   # File upload
   uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt'])

   if uploaded_file is not None:
       # Initialize SemantiCore
       core = SemantiCore()
       
       # Process the file
       with st.spinner("Processing document..."):
           result = core.process_document(uploaded_file)
       
       # Display results
       st.success("Processing complete!")
       
       col1, col2, col3 = st.columns(3)
       
       with col1:
           st.metric("Entities", len(result.entities))
       
       with col2:
           st.metric("Triples", len(result.triples))
       
       with col3:
           st.metric("Embeddings", len(result.embeddings))
       
       # Show entities
       st.subheader("Extracted Entities")
       for entity in result.entities[:10]:
           st.write(f"• {entity.text} ({entity.type})")

Performance Optimization Examples
--------------------------------

**Batch Processing for Large Datasets**

.. code-block:: python

   from semanticore import SemantiCore
   import concurrent.futures

   # Initialize SemantiCore with optimized settings
   core = SemantiCore(
       config={
           "processing": {
               "batch_size": 100,
               "max_workers": 8,
               "timeout": 600,
               "memory_limit": "8GB"
           }
       }
   )

   # Process large dataset in batches
   documents = load_large_dataset()  # 10,000+ documents

   def process_batch(batch):
       return core.process_documents(batch)

   # Split into batches
   batch_size = 100
   batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

   # Process batches in parallel
   with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
       results = list(executor.map(process_batch, batches))

   print(f"Processed {len(documents)} documents in {len(batches)} batches")

**Memory-Efficient Processing**

.. code-block:: python

   from semanticore import SemantiCore
   import gc

   # Initialize with memory optimization
   core = SemantiCore(
       config={
           "processing": {
               "batch_size": 50,
               "max_workers": 2,
               "memory_limit": "4GB",
               "cleanup_interval": 10
           }
       }
   )

   # Process documents with memory management
   documents = load_documents()

   for i, doc in enumerate(documents):
       # Process single document
       result = core.process_document(doc)
       
       # Store results
       store_results(result)
       
       # Clean up memory periodically
       if i % 10 == 0:
           gc.collect()
           print(f"Processed {i} documents, memory cleaned")

.. raw:: html

   <div style="text-align: center; margin: 20px 0; padding: 15px; background-color: #f0f8ff; border-left: 4px solid #27AE60; border-radius: 5px;">
       <strong>💡 Tip:</strong> These examples can be combined and customized for your specific use case. Check the API reference for more detailed parameter options.
   </div> 