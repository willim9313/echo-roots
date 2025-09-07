"""
Database migration system for echo-roots storage.

This module provides schema versioning and migration capabilities
to support evolution of the storage layer over time while maintaining
data integrity and backward compatibility.

Features:
- Version-controlled schema changes
- Forward and backward migrations
- Data preservation during schema updates
- Migration rollback capabilities
- Environment-specific configurations
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import asyncio

from .duckdb_backend import DuckDBBackend
from .interfaces import StorageError


logger = logging.getLogger(__name__)


class Migration:
    """Represents a single database migration."""
    
    def __init__(
        self,
        version: str,
        description: str,
        up_sql: str,
        down_sql: Optional[str] = None,
        pre_migration: Optional[Callable] = None,
        post_migration: Optional[Callable] = None
    ):
        self.version = version
        self.description = description
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.pre_migration = pre_migration
        self.post_migration = post_migration
    
    def __str__(self) -> str:
        return f"Migration {self.version}: {self.description}"


class MigrationManager:
    """Manages database schema migrations."""
    
    def __init__(self, backend: DuckDBBackend):
        self.backend = backend
        self.migrations: List[Migration] = []
        self._register_migrations()
    
    def _register_migrations(self):
        """Register all available migrations."""
        
        # Initial schema - v1.0.0
        self.migrations.append(Migration(
            version="1.0.0",
            description="Initial schema for echo-roots taxonomy system",
            up_sql=self._get_initial_schema(),
            down_sql="DROP SCHEMA IF EXISTS main CASCADE;"
        ))
        
        # Future migrations will be added here
        # Example:
        # self.migrations.append(Migration(
        #     version="1.1.0", 
        #     description="Add indexes for performance optimization",
        #     up_sql="CREATE INDEX ...",
        #     down_sql="DROP INDEX ..."
        # ))
    
    def _get_initial_schema(self) -> str:
        """Get the initial database schema."""
        return """
        -- Create migration tracking table
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rollback_sql TEXT
        );
        
        -- Ingestion items table  
        CREATE TABLE IF NOT EXISTS ingestion_items (
            id VARCHAR PRIMARY KEY,
            title VARCHAR NOT NULL,
            description TEXT,
            raw_category VARCHAR,
            raw_attributes JSON,
            source VARCHAR NOT NULL,
            language VARCHAR DEFAULT 'auto',
            metadata JSON,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Extraction results table
        CREATE TABLE IF NOT EXISTS extraction_results (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            item_id VARCHAR NOT NULL,
            attributes JSON NOT NULL,
            terms JSON NOT NULL,
            extraction_metadata JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (item_id) REFERENCES ingestion_items(id)
        );
        
        -- Categories table
        CREATE TABLE IF NOT EXISTS categories (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            name VARCHAR NOT NULL,
            description TEXT,
            domain VARCHAR NOT NULL,
            parent_id VARCHAR,
            level INTEGER DEFAULT 0,
            category_metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_id) REFERENCES categories(id),
            UNIQUE(name, domain, parent_id)
        );
        
        -- Attributes table
        CREATE TABLE IF NOT EXISTS attributes (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            name VARCHAR NOT NULL,
            data_type VARCHAR NOT NULL,
            description TEXT,
            category_id VARCHAR NOT NULL,
            is_required BOOLEAN DEFAULT FALSE,
            attribute_metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (category_id) REFERENCES categories(id),
            UNIQUE(name, category_id)
        );
        
        -- Semantic candidates table
        CREATE TABLE IF NOT EXISTS semantic_candidates (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            term VARCHAR NOT NULL,
            normalized_term VARCHAR NOT NULL,
            frequency INTEGER DEFAULT 1,
            contexts JSON,
            cluster_id VARCHAR,
            score DOUBLE DEFAULT 0.0,
            language VARCHAR DEFAULT 'auto',
            domain VARCHAR NOT NULL,
            category_id VARCHAR,
            status VARCHAR DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            candidate_metadata JSON,
            FOREIGN KEY (category_id) REFERENCES categories(id)
        );
        
        -- Mappings table
        CREATE TABLE IF NOT EXISTS mappings (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            name VARCHAR NOT NULL,
            domain VARCHAR NOT NULL,
            source_schema JSON NOT NULL,
            target_schema JSON NOT NULL,
            field_mappings JSON NOT NULL,
            transformation_rules JSON,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, domain)
        );
        
        -- Elevation proposals table
        CREATE TABLE IF NOT EXISTS elevation_proposals (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            proposal_type VARCHAR NOT NULL,
            domain VARCHAR NOT NULL,
            source_item_id VARCHAR,
            proposed_changes JSON NOT NULL,
            justification TEXT,
            status VARCHAR DEFAULT 'pending',
            reviewer_id VARCHAR,
            reviewer_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_item_id) REFERENCES ingestion_items(id)
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_ingestion_source ON ingestion_items(source);
        CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_items(status);
        CREATE INDEX IF NOT EXISTS idx_ingestion_created ON ingestion_items(created_at);
        CREATE INDEX IF NOT EXISTS idx_ingestion_language ON ingestion_items(language);
        
        CREATE INDEX IF NOT EXISTS idx_extraction_item ON extraction_results(item_id);
        CREATE INDEX IF NOT EXISTS idx_extraction_created ON extraction_results(created_at);
        
        CREATE INDEX IF NOT EXISTS idx_categories_domain ON categories(domain);
        CREATE INDEX IF NOT EXISTS idx_categories_parent ON categories(parent_id);
        CREATE INDEX IF NOT EXISTS idx_categories_level ON categories(level);
        
        CREATE INDEX IF NOT EXISTS idx_attributes_category ON attributes(category_id);
        
        CREATE INDEX IF NOT EXISTS idx_semantic_domain ON semantic_candidates(domain);
        CREATE INDEX IF NOT EXISTS idx_semantic_category ON semantic_candidates(category_id);
        CREATE INDEX IF NOT EXISTS idx_semantic_term ON semantic_candidates(normalized_term);
        CREATE INDEX IF NOT EXISTS idx_semantic_cluster ON semantic_candidates(cluster_id);
        
        CREATE INDEX IF NOT EXISTS idx_mappings_domain ON mappings(domain);
        CREATE INDEX IF NOT EXISTS idx_mappings_active ON mappings(is_active);
        
        CREATE INDEX IF NOT EXISTS idx_proposals_status ON elevation_proposals(status);
        CREATE INDEX IF NOT EXISTS idx_proposals_domain ON elevation_proposals(domain);
        CREATE INDEX IF NOT EXISTS idx_proposals_created ON elevation_proposals(created_at);
        """
    
    async def get_current_version(self) -> Optional[str]:
        """Get the current schema version."""
        if not self.backend.connection:
            raise StorageError("Database not initialized")
        
        try:
            # Check if migrations table exists
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.backend.connection.execute,
                """
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = 'schema_migrations'
                """
            )
            
            if result.fetchone()[0] == 0:
                return None
            
            # Get latest migration
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.backend.connection.execute,
                "SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1"
            )
            
            row = result.fetchone()
            return row[0] if row else None
            
        except Exception as e:
            logger.error(f"Failed to get current version: {e}")
            return None
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        if not self.backend.connection:
            raise StorageError("Database not initialized")
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.backend.connection.execute,
                "SELECT version FROM schema_migrations ORDER BY applied_at"
            )
            
            return [row[0] for row in result.fetchall()]
            
        except Exception:
            # If table doesn't exist, no migrations applied
            return []
    
    async def apply_migration(self, migration: Migration) -> None:
        """Apply a single migration."""
        if not self.backend.connection:
            raise StorageError("Database not initialized")
        
        logger.info(f"Applying migration: {migration}")
        
        try:
            # Start transaction
            async with self.backend.transaction():
                # Run pre-migration hook
                if migration.pre_migration:
                    await migration.pre_migration(self.backend)
                
                # Execute migration SQL
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.backend.connection.execute,
                    migration.up_sql
                )
                
                # Record migration
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.backend.connection.execute,
                    """
                    INSERT INTO schema_migrations (version, description, rollback_sql)
                    VALUES (?, ?, ?)
                    """,
                    [migration.version, migration.description, migration.down_sql]
                )
                
                # Run post-migration hook
                if migration.post_migration:
                    await migration.post_migration(self.backend)
            
            logger.info(f"Successfully applied migration {migration.version}")
            
        except Exception as e:
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            raise StorageError(f"Migration failed: {e}")
    
    async def rollback_migration(self, version: str) -> None:
        """Rollback a specific migration."""
        if not self.backend.connection:
            raise StorageError("Database not initialized")
        
        logger.info(f"Rolling back migration: {version}")
        
        try:
            # Get migration info
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.backend.connection.execute,
                "SELECT rollback_sql FROM schema_migrations WHERE version = ?",
                [version]
            )
            
            row = result.fetchone()
            if not row or not row[0]:
                raise StorageError(f"No rollback available for migration {version}")
            
            rollback_sql = row[0]
            
            # Start transaction
            async with self.backend.transaction():
                # Execute rollback SQL
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.backend.connection.execute,
                    rollback_sql
                )
                
                # Remove migration record
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.backend.connection.execute,
                    "DELETE FROM schema_migrations WHERE version = ?",
                    [version]
                )
            
            logger.info(f"Successfully rolled back migration {version}")
            
        except Exception as e:
            logger.error(f"Failed to rollback migration {version}: {e}")
            raise StorageError(f"Rollback failed: {e}")
    
    async def migrate_to_latest(self) -> None:
        """Migrate database to the latest version."""
        current_version = await self.get_current_version()
        applied_migrations = await self.get_applied_migrations()
        
        logger.info(f"Current schema version: {current_version or 'None'}")
        
        # Find migrations to apply
        migrations_to_apply = []
        for migration in self.migrations:
            if migration.version not in applied_migrations:
                migrations_to_apply.append(migration)
        
        if not migrations_to_apply:
            logger.info("Database is already up to date")
            return
        
        # Sort migrations by version
        migrations_to_apply.sort(key=lambda m: m.version)
        
        logger.info(f"Applying {len(migrations_to_apply)} migrations")
        
        for migration in migrations_to_apply:
            await self.apply_migration(migration)
        
        final_version = await self.get_current_version()
        logger.info(f"Migration complete. Current version: {final_version}")
    
    async def migrate_to_version(self, target_version: str) -> None:
        """Migrate database to a specific version."""
        current_version = await self.get_current_version()
        applied_migrations = await self.get_applied_migrations()
        
        # Find target migration
        target_migration = next(
            (m for m in self.migrations if m.version == target_version), 
            None
        )
        
        if not target_migration:
            raise StorageError(f"Migration version {target_version} not found")
        
        # Determine if we need to go forward or backward
        if target_version in applied_migrations:
            logger.info(f"Already at version {target_version}")
            return
        
        # For now, only support forward migrations
        # TODO: Implement proper rollback logic for backward migrations
        migrations_to_apply = []
        for migration in self.migrations:
            if (migration.version not in applied_migrations and 
                migration.version <= target_version):
                migrations_to_apply.append(migration)
        
        # Sort migrations by version
        migrations_to_apply.sort(key=lambda m: m.version)
        
        logger.info(f"Applying {len(migrations_to_apply)} migrations to reach {target_version}")
        
        for migration in migrations_to_apply:
            await self.apply_migration(migration)
            if migration.version == target_version:
                break
        
        logger.info(f"Migration to {target_version} complete")
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get detailed migration status."""
        current_version = await self.get_current_version()
        applied_migrations = await self.get_applied_migrations()
        
        available_migrations = [m.version for m in self.migrations]
        pending_migrations = [
            v for v in available_migrations 
            if v not in applied_migrations
        ]
        
        return {
            "current_version": current_version,
            "applied_migrations": applied_migrations,
            "available_migrations": available_migrations,
            "pending_migrations": pending_migrations,
            "total_migrations": len(self.migrations),
            "applied_count": len(applied_migrations),
            "pending_count": len(pending_migrations)
        }
    
    def add_migration(self, migration: Migration) -> None:
        """Add a new migration to the manager."""
        # Check for version conflicts
        existing_versions = [m.version for m in self.migrations]
        if migration.version in existing_versions:
            raise ValueError(f"Migration version {migration.version} already exists")
        
        self.migrations.append(migration)
        # Sort migrations by version
        self.migrations.sort(key=lambda m: m.version)
        
        logger.info(f"Added migration: {migration}")


async def initialize_database(backend: DuckDBBackend) -> None:
    """Initialize database with latest schema."""
    migration_manager = MigrationManager(backend)
    await migration_manager.migrate_to_latest()


async def get_migration_manager(backend: DuckDBBackend) -> MigrationManager:
    """Get a configured migration manager."""
    return MigrationManager(backend)
