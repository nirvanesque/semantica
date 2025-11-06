"""
Data Ingestion Module

This module provides comprehensive data ingestion capabilities from various sources.

Exports:
    - FileIngestor: Local and cloud file processing
    - WebIngestor: Web scraping and crawling
    - FeedIngestor: RSS/Atom feed processing
    - StreamIngestor: Real-time stream processing
    - RepoIngestor: Git repository processing
    - EmailIngestor: Email protocol handling
    - DBIngestor: Database export handling
    - build: Module-level build function for data ingestion
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .file_ingestor import FileIngestor, FileObject, FileTypeDetector, CloudStorageIngestor
from .web_ingestor import WebIngestor, WebContent, RateLimiter, RobotsChecker, ContentExtractor, SitemapCrawler
from .feed_ingestor import FeedIngestor, FeedItem, FeedData, FeedParser, FeedMonitor
from .stream_ingestor import (
    StreamIngestor,
    StreamMessage,
    StreamProcessor,
    KafkaProcessor,
    RabbitMQProcessor,
    KinesisProcessor,
    PulsarProcessor,
    StreamMonitor,
)
from .repo_ingestor import RepoIngestor, CodeFile, CommitInfo, CodeExtractor, GitAnalyzer
from .email_ingestor import EmailIngestor, EmailData, AttachmentProcessor, EmailParser as EmailIngestorParser
from .db_ingestor import DBIngestor, TableData, DatabaseConnector, DataExporter

__all__ = [
    # File ingestion
    "FileIngestor",
    "FileObject",
    "FileTypeDetector",
    "CloudStorageIngestor",
    # Web ingestion
    "WebIngestor",
    "WebContent",
    "RateLimiter",
    "RobotsChecker",
    "ContentExtractor",
    "SitemapCrawler",
    # Feed ingestion
    "FeedIngestor",
    "FeedItem",
    "FeedData",
    "FeedParser",
    "FeedMonitor",
    # Stream ingestion
    "StreamIngestor",
    "StreamMessage",
    "StreamProcessor",
    "KafkaProcessor",
    "RabbitMQProcessor",
    "KinesisProcessor",
    "PulsarProcessor",
    "StreamMonitor",
    # Repository ingestion
    "RepoIngestor",
    "CodeFile",
    "CommitInfo",
    "CodeExtractor",
    "GitAnalyzer",
    # Email ingestion
    "EmailIngestor",
    "EmailData",
    "AttachmentProcessor",
    "EmailIngestorParser",
    # Database ingestion
    "DBIngestor",
    "TableData",
    "DatabaseConnector",
    "DataExporter",
    "build",
]


def build(
    sources: Union[List[Union[str, Path]], str, Path],
    source_type: str = "file",
    recursive: bool = True,
    read_content: bool = True,
    **options
) -> Dict[str, Any]:
    """
    Ingest data from sources (module-level convenience function).
    
    This is a user-friendly wrapper that automatically selects the appropriate
    ingestor based on source type and ingests the data.
    
    Args:
        sources: Data source(s) - can be file paths, URLs, directories, etc.
        source_type: Type of source - "file", "web", "feed", "stream", "repo", "email", "db" (default: "file")
        recursive: For directories, whether to ingest recursively (default: True)
        read_content: Whether to read file content (default: True)
        **options: Additional ingestion options
        
    Returns:
        Dictionary containing:
            - files: List of ingested file objects (for file ingestion)
            - content: Ingested content (for web/feed ingestion)
            - metadata: Ingestion metadata
            - statistics: Ingestion statistics
            
    Examples:
        >>> import semantica
        >>> result = semantica.ingest.build(
        ...     sources=["doc1.pdf", "doc2.docx"],
        ...     source_type="file",
        ...     read_content=True
        ... )
        >>> print(f"Ingested {len(result['files'])} files")
    """
    # Normalize sources to list
    if isinstance(sources, (str, Path)):
        sources = [sources]
    
    results = {
        "files": [],
        "content": [],
        "metadata": {},
        "statistics": {}
    }
    
    if source_type == "file":
        # Use FileIngestor
        ingestor = FileIngestor(config=options.get("config", {}), **options)
        
        file_objects = []
        for source in sources:
            source_path = Path(source)
            if source_path.is_dir():
                # Ingest directory
                files = ingestor.ingest_directory(source_path, recursive=recursive, **options)
                file_objects.extend(files)
            elif source_path.is_file():
                # Ingest single file
                file_obj = ingestor.ingest_file(source_path, read_content=read_content, **options)
                file_objects.append(file_obj)
            else:
                # Try as file path string
                try:
                    file_obj = ingestor.ingest_file(source, read_content=read_content, **options)
                    file_objects.append(file_obj)
                except Exception as e:
                    results["statistics"].setdefault("errors", []).append({
                        "source": str(source),
                        "error": str(e)
                    })
        
        results["files"] = file_objects
        results["statistics"] = {
            "total_sources": len(sources),
            "ingested_files": len(file_objects),
            "errors": len(results["statistics"].get("errors", []))
        }
        
    elif source_type == "web":
        # Use WebIngestor
        from .web_ingestor import WebIngestor
        ingestor = WebIngestor(config=options.get("config", {}), **options)
        
        web_contents = []
        for source in sources:
            if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
                content = ingestor.ingest_url(source, **options)
                web_contents.append(content)
        
        results["content"] = web_contents
        results["statistics"] = {
            "total_urls": len(sources),
            "ingested_pages": len(web_contents)
        }
        
    else:
        # For other types, return placeholder
        results["statistics"] = {
            "message": f"Source type '{source_type}' ingestion not yet implemented in build() function",
            "suggestion": f"Use {source_type.capitalize()}Ingestor class directly"
        }
    
    return results
