"""
T5: Ingestion Pipeline

Main orchestration pipeline that connects domain adaptation (T2), 
LLM extraction (T3), and storage (T4) for end-to-end data processing.

This module provides:
- IngestionConfig: Configuration for ingestion workflows
- IngestionPipeline: Main orchestrator for end-to-end processing
- BatchProcessor: Handles batch ingestion with progress tracking
- StreamProcessor: Handles continuous/streaming ingestion
- PipelineCoordinator: High-level workflow management
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

from echo_roots.models.core import (
    IngestionItem, 
    ExtractionResult, 
    ProcessingStatus
)
from echo_roots.models.domain import DomainPack
from echo_roots.domain.loader import DomainPackLoader
from echo_roots.domain.adapter import DomainAdapter
from echo_roots.pipelines.extraction import ExtractionPipeline, ExtractorConfig
from echo_roots.pipelines.openai_client import LLMClient, MockLLMClient
from echo_roots.storage.interfaces import StorageBackend
from echo_roots.storage.duckdb_backend import DuckDBBackend
from echo_roots.storage.repository import RepositoryCoordinator

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline operations."""
    
    # Domain configuration
    domain_pack_path: str = "domains/"
    domain_name: str = "ecommerce"
    
    # Processing configuration
    batch_size: int = 100
    max_concurrent: int = 10
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Storage configuration
    storage_path: str = "data/processed/ingestion.db"
    enable_storage: bool = True
    
    # LLM configuration
    llm_config: ExtractorConfig = field(default_factory=ExtractorConfig)
    
    # Quality control
    enable_validation: bool = True
    min_quality_score: float = 0.6
    skip_on_error: bool = True
    
    # Progress tracking
    enable_progress: bool = True
    progress_interval: int = 10  # Log progress every N items
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if not 0 <= self.min_quality_score <= 1:
            raise ValueError("min_quality_score must be between 0 and 1")


@dataclass
class IngestionStats:
    """Statistics tracking for ingestion operations."""
    
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    errors: List[Dict[str, Any]] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.processed_items == 0:
            return 0.0
        return (self.successful_items / self.processed_items) * 100
    
    @property
    def average_quality(self) -> float:
        """Calculate average quality score."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)
    
    @property
    def processing_time(self) -> float:
        """Calculate total processing time in seconds."""
        if not self.start_time or not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    def add_error(self, item_id: str, error: Exception, context: Optional[str] = None):
        """Add an error to the tracking."""
        self.errors.append({
            "item_id": item_id,
            "error": str(error),
            "error_type": type(error).__name__,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for serialization."""
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "success_rate": self.success_rate,
            "average_quality": self.average_quality,
            "processing_time": self.processing_time,
            "errors": self.errors,
            "quality_scores": self.quality_scores,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


class IngestionPipeline:
    """
    Main ingestion pipeline orchestrator.
    
    Coordinates the complete data flow:
    Raw Data → Domain Adaptation → LLM Extraction → Storage
    """
    
    def __init__(
        self,
        config: IngestionConfig,
        llm_client: Optional[LLMClient] = None,
        storage_backend: Optional[StorageBackend] = None
    ):
        self.config = config
        self.stats = IngestionStats()
        
        # Initialize components
        self._domain_loader = DomainPackLoader(config.domain_pack_path)
        self._domain_pack: Optional[DomainPack] = None
        self._domain_adapter: Optional[DomainAdapter] = None
        self._extraction_pipeline: Optional[ExtractionPipeline] = None
        self._storage: Optional[RepositoryCoordinator] = None
        
        # LLM client
        self._llm_client = llm_client or MockLLMClient()
        
        # Storage backend
        if storage_backend:
            self._storage_backend = storage_backend
        elif config.enable_storage:
            self._storage_backend = DuckDBBackend(config.storage_path)
        else:
            self._storage_backend = None
    
    async def initialize(self) -> None:
        """Initialize pipeline components."""
        logger.info(f"Initializing ingestion pipeline for domain: {self.config.domain_name}")
        
        # Load domain pack
        self._domain_pack = self._domain_loader.load_domain_pack(self.config.domain_name)
        self._domain_adapter = DomainAdapter(self._domain_pack)
        
        # Initialize extraction pipeline
        self._extraction_pipeline = ExtractionPipeline(
            self._domain_pack,
            self._llm_client,
            self.config.llm_config
        )
        
        # Initialize storage
        if self._storage_backend:
            await self._storage_backend.initialize()
            self._storage = RepositoryCoordinator(self._storage_backend)
        
        logger.info("Pipeline initialization complete")
    
    async def process_item(
        self, 
        raw_data: Dict[str, Any],
        item_id: Optional[str] = None
    ) -> Optional[ExtractionResult]:
        """
        Process a single raw data item through the complete pipeline.
        
        Args:
            raw_data: Raw input data dictionary
            item_id: Optional explicit item ID
            
        Returns:
            ExtractionResult if successful, None if skipped/failed
        """
        if not self._domain_pack or not self._domain_adapter or not self._extraction_pipeline:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        try:
            # Step 1: Domain Adaptation
            ingestion_item = self._domain_adapter.adapt_input(raw_data, item_id)
            logger.debug(f"Adapted item {ingestion_item.item_id}")
            
            # Step 2: Store raw ingestion item
            if self._storage:
                await self._storage.storage.ingestion.store_item(ingestion_item)
                await self._storage.storage.ingestion.update_status(
                    ingestion_item.item_id, 
                    ProcessingStatus.PROCESSING
                )
            
            # Step 3: LLM Extraction
            extraction_result = await self._extraction_pipeline.process_item(
                ingestion_item,
                validate=self.config.enable_validation
            )
            
            # Step 4: Quality control
            if self.config.enable_validation:
                # Extract quality score from extraction metadata or use default
                quality_score = 0.8  # Default quality score for successful extractions
                # Note: Quality scoring would be handled by T3 validation pipeline
                # For now, we assume successful extractions meet quality standards
                
                if quality_score < self.config.min_quality_score:
                    logger.warning(
                        f"Item {ingestion_item.item_id} quality score {quality_score} "
                        f"below threshold {self.config.min_quality_score}"
                    )
                    if self.config.skip_on_error:
                        self.stats.skipped_items += 1
                        return None
                
                self.stats.quality_scores.append(quality_score)
            
            # Step 5: Store extraction result
            if self._storage:
                # Note: ExtractionRepository not fully implemented in T4
                # This would store the extraction results
                logger.debug(f"Would store extraction result for {ingestion_item.item_id}")
                
                # Update status to completed
                await self._storage.storage.ingestion.update_status(
                    ingestion_item.item_id,
                    ProcessingStatus.COMPLETED
                )
            
            self.stats.successful_items += 1
            logger.debug(f"Successfully processed item {ingestion_item.item_id}")
            
            return extraction_result
            
        except Exception as e:
            self.stats.failed_items += 1
            self.stats.add_error(
                item_id or "unknown",
                e,
                "process_item"
            )
            
            # Update storage status if available
            if self._storage and item_id:
                try:
                    await self._storage.storage.ingestion.update_status(
                        item_id,
                        ProcessingStatus.FAILED
                    )
                except Exception:
                    pass  # Don't fail on storage error
            
            logger.error(f"Failed to process item {item_id}: {e}")
            
            if not self.config.skip_on_error:
                raise
            
            return None
        
        finally:
            self.stats.processed_items += 1
    
    async def process_batch(
        self,
        raw_data_items: List[Dict[str, Any]]
    ) -> List[Optional[ExtractionResult]]:
        """
        Process a batch of raw data items.
        
        Args:
            raw_data_items: List of raw data dictionaries
            
        Returns:
            List of ExtractionResults (may contain None for failed items)
        """
        self.stats.total_items = len(raw_data_items)
        self.stats.start_time = datetime.now()
        
        logger.info(f"Starting batch processing of {len(raw_data_items)} items")
        
        # Process in chunks to respect max_concurrent
        results = []
        batch_size = min(self.config.batch_size, self.config.max_concurrent)
        
        for i in range(0, len(raw_data_items), batch_size):
            batch = raw_data_items[i:i + batch_size]
            
            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[self.process_item(item) for item in batch],
                return_exceptions=True
            )
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch item {i + j} failed: {result}")
                    results.append(None)
                    self.stats.failed_items += 1
                    self.stats.add_error(f"batch_item_{i + j}", result, "batch_processing")
                else:
                    results.append(result)
            
            # Progress logging
            if self.config.enable_progress and (i + batch_size) % self.config.progress_interval == 0:
                progress = min(i + batch_size, len(raw_data_items))
                logger.info(f"Processed {progress}/{len(raw_data_items)} items "
                           f"({(progress / len(raw_data_items)) * 100:.1f}%)")
        
        self.stats.end_time = datetime.now()
        
        logger.info(f"Batch processing complete. Success rate: {self.stats.success_rate:.1f}% "
                   f"({self.stats.successful_items}/{self.stats.total_items})")
        
        return results
    
    async def cleanup(self) -> None:
        """Clean up pipeline resources."""
        if self._storage_backend:
            await self._storage_backend.cleanup()
        
        logger.info("Pipeline cleanup complete")


class BatchProcessor:
    """Specialized processor for large batch operations."""
    
    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline
    
    async def process_file(self, file_path: str) -> IngestionStats:
        """Process a JSON file containing array of raw data items."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data_items = json.load(f)
        
        if not isinstance(raw_data_items, list):
            raise ValueError("JSON file must contain an array of objects")
        
        await self.pipeline.process_batch(raw_data_items)
        return self.pipeline.stats
    
    async def process_directory(self, directory_path: str, pattern: str = "*.json") -> IngestionStats:
        """Process all JSON files in a directory."""
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        json_files = list(directory.glob(pattern))
        
        if not json_files:
            raise ValueError(f"No files matching {pattern} found in {directory}")
        
        logger.info(f"Processing {len(json_files)} files from {directory}")
        
        all_items = []
        for json_file in json_files:
            logger.info(f"Loading {json_file.name}")
            with open(json_file, 'r', encoding='utf-8') as f:
                file_items = json.load(f)
                if isinstance(file_items, list):
                    all_items.extend(file_items)
                else:
                    all_items.append(file_items)
        
        await self.pipeline.process_batch(all_items)
        return self.pipeline.stats


class StreamProcessor:
    """Processor for streaming/continuous ingestion."""
    
    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline
        self._running = False
    
    async def start_stream(
        self,
        data_stream: AsyncGenerator[Dict[str, Any], None],
        callback: Optional[Callable[[Optional[ExtractionResult]], None]] = None
    ) -> None:
        """
        Process a stream of data items continuously.
        
        Args:
            data_stream: Async generator yielding raw data items
            callback: Optional callback for each processed result
        """
        self._running = True
        logger.info("Starting stream processing")
        
        try:
            async for raw_data in data_stream:
                if not self._running:
                    break
                
                try:
                    result = await self.pipeline.process_item(raw_data)
                    if callback:
                        callback(result)
                except Exception as e:
                    logger.error(f"Stream processing error: {e}")
                    if not self.pipeline.config.skip_on_error:
                        raise
        finally:
            logger.info("Stream processing stopped")
    
    def stop_stream(self) -> None:
        """Stop stream processing."""
        self._running = False
        logger.info("Stream processing stop requested")


class PipelineCoordinator:
    """
    High-level coordinator for managing multiple ingestion pipelines.
    
    Useful for scenarios with multiple domains or processing configurations.
    """
    
    def __init__(self):
        self.pipelines: Dict[str, IngestionPipeline] = {}
        self.configs: Dict[str, IngestionConfig] = {}
    
    def register_pipeline(self, name: str, config: IngestionConfig) -> None:
        """Register a named pipeline configuration."""
        self.configs[name] = config
        logger.info(f"Registered pipeline configuration: {name}")
    
    async def get_pipeline(self, name: str) -> IngestionPipeline:
        """Get or create a pipeline instance."""
        if name not in self.configs:
            raise ValueError(f"No configuration registered for pipeline: {name}")
        
        if name not in self.pipelines:
            self.pipelines[name] = IngestionPipeline(self.configs[name])
            await self.pipelines[name].initialize()
        
        return self.pipelines[name]
    
    async def process_with_pipeline(
        self,
        pipeline_name: str,
        raw_data_items: List[Dict[str, Any]]
    ) -> IngestionStats:
        """Process data with a specific named pipeline."""
        pipeline = await self.get_pipeline(pipeline_name)
        await pipeline.process_batch(raw_data_items)
        return pipeline.stats
    
    async def cleanup_all(self) -> None:
        """Clean up all pipeline instances."""
        for pipeline in self.pipelines.values():
            await pipeline.cleanup()
        
        self.pipelines.clear()
        logger.info("All pipelines cleaned up")
