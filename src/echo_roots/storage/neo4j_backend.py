"""
Neo4j storage backend implementation.

This module provides the Neo4j-based storage implementation for 
taxonomy trees, controlled vocabulary, and semantic relationships.
Optimized for graph queries, hierarchy navigation, and relationship management.

Features:
- Taxonomy tree operations (A layer)
- Controlled vocabulary management (C layer)  
- Semantic relationship handling
- Graph traversal and pathfinding
- Multi-language support
- Transaction management
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4

import neo4j
from neo4j import AsyncGraphDatabase, AsyncSession
from pydantic import ValidationError

from .interfaces import (
    StorageBackend, TaxonomyRepository, TransactionContext,
    StorageError, ConnectionError, IntegrityError, NotFoundError, ConflictError
)
from ..models.taxonomy import Category, Attribute, AttributeValue, SemanticCandidate, SemanticRelation


logger = logging.getLogger(__name__)


class Neo4jTransaction(TransactionContext):
    """Transaction context for Neo4j operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.transaction = None
    
    async def __aenter__(self):
        self.transaction = await self.session.begin_transaction()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.transaction:
            if exc_type is None:
                await self.commit()
            else:
                await self.rollback()
    
    async def commit(self) -> None:
        """Commit the current transaction."""
        if self.transaction:
            await self.transaction.commit()
            self.transaction = None
    
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        if self.transaction:
            await self.transaction.rollback()
            self.transaction = None


class Neo4jBackend:
    """Core Neo4j storage backend."""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user  
        self.password = password
        self.database = database
        self.driver = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Neo4j connection and create constraints."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            
            # Verify connectivity
            await self.driver.verify_connectivity()
            
            # Create constraints and indexes
            await self._create_constraints_and_indexes()
            
            self._initialized = True
            logger.info(f"Neo4j backend initialized: {self.uri}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Neo4j: {e}")
    
    async def _create_constraints_and_indexes(self) -> None:
        """Create necessary constraints and indexes."""
        constraints_and_indexes = [
            # Unique constraints
            "CREATE CONSTRAINT category_id_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.category_id IS UNIQUE",
            "CREATE CONSTRAINT attribute_id_unique IF NOT EXISTS FOR (a:Attribute) REQUIRE a.attribute_id IS UNIQUE",
            "CREATE CONSTRAINT candidate_id_unique IF NOT EXISTS FOR (s:SemanticCandidate) REQUIRE s.candidate_id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX category_name_idx IF NOT EXISTS FOR (c:Category) ON (c.name)",
            "CREATE INDEX category_level_idx IF NOT EXISTS FOR (c:Category) ON (c.level)",
            "CREATE INDEX attribute_name_idx IF NOT EXISTS FOR (a:Attribute) ON (a.name)",
            "CREATE INDEX candidate_term_idx IF NOT EXISTS FOR (s:SemanticCandidate) ON (s.normalized_term)",
            "CREATE INDEX candidate_status_idx IF NOT EXISTS FOR (s:SemanticCandidate) ON (s.status)",
        ]
        
        async with self.driver.session(database=self.database) as session:
            for constraint in constraints_and_indexes:
                try:
                    await session.run(constraint)
                except neo4j.exceptions.ClientError as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Failed to create constraint/index: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health and return metrics."""
        if not self._initialized or not self.driver:
            return {"status": "unhealthy", "error": "Not initialized"}
        
        try:
            async with self.driver.session(database=self.database) as session:
                # Test basic connectivity
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                
                # Get node counts
                stats_query = """
                CALL apoc.meta.stats() YIELD labels, relTypesCount
                RETURN labels, relTypesCount
                """
                
                try:
                    stats_result = await session.run(stats_query)
                    stats_record = await stats_result.single()
                    node_counts = stats_record["labels"] if stats_record else {}
                    rel_counts = stats_record["relTypesCount"] if stats_record else {}
                except:
                    # Fallback if APOC is not available
                    node_counts = {}
                    rel_counts = {}
                
                return {
                    "status": "healthy",
                    "database": self.database,
                    "uri": self.uri,
                    "node_counts": node_counts,
                    "relationship_counts": rel_counts,
                    "connection_test": "passed"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            self._initialized = False
            logger.info("Neo4j connection closed")
    
    def session(self) -> AsyncSession:
        """Create a new session."""
        if not self.driver:
            raise ConnectionError("Database not initialized")
        return self.driver.session(database=self.database)


class Neo4jTaxonomyRepository:
    """Neo4j implementation of TaxonomyRepository."""
    
    def __init__(self, backend: Neo4jBackend):
        self.backend = backend
    
    async def store_category(self, category: Category) -> str:
        """Store a category node and its relationships."""
        if not self.backend.driver:
            raise ConnectionError("Database not initialized")
        
        try:
            async with self.backend.session() as session:
                # Create category node
                create_query = """
                MERGE (c:Category {category_id: $category_id})
                SET c += $properties
                RETURN c.category_id as id
                """
                
                properties = {
                    "name": category.name,
                    "level": category.level,
                    "path": category.path,
                    "labels": json.dumps(category.labels) if category.labels else "{}",
                    "description": category.description,
                    "status": category.status,
                    "created_at": category.created_at.isoformat(),
                    "updated_at": category.updated_at.isoformat(),
                    "metadata": json.dumps(category.metadata) if category.metadata else "{}"
                }
                
                result = await session.run(create_query, {
                    "category_id": category.category_id,
                    "properties": properties
                })
                
                record = await result.single()
                if not record:
                    raise StorageError("Failed to create category")
                
                # Create parent-child relationship if parent exists
                if category.parent_id:
                    parent_rel_query = """
                    MATCH (parent:Category {category_id: $parent_id})
                    MATCH (child:Category {category_id: $child_id})
                    MERGE (parent)-[:HAS_CHILD]->(child)
                    MERGE (child)-[:CHILD_OF]->(parent)
                    """
                    
                    await session.run(parent_rel_query, {
                        "parent_id": category.parent_id,
                        "child_id": category.category_id
                    })
                
                return category.category_id
                
        except Exception as e:
            logger.error(f"Failed to store category: {e}")
            raise StorageError(f"Failed to store category: {e}")
    
    async def get_category(self, category_id: str) -> Optional[Category]:
        """Retrieve a category by ID."""
        if not self.backend.driver:
            raise ConnectionError("Database not initialized")
        
        try:
            async with self.backend.session() as session:
                query = """
                MATCH (c:Category {category_id: $category_id})
                RETURN c
                """
                
                result = await session.run(query, {"category_id": category_id})
                record = await result.single()
                
                if not record:
                    return None
                
                node = record["c"]
                
                return Category(
                    category_id=node["category_id"],
                    name=node["name"],
                    parent_id=node.get("parent_id"),
                    level=node["level"],
                    path=node["path"],
                    labels=json.loads(node.get("labels", "{}")),
                    description=node.get("description"),
                    status=node["status"],
                    created_at=datetime.fromisoformat(node["created_at"]),
                    updated_at=datetime.fromisoformat(node["updated_at"]),
                    metadata=json.loads(node.get("metadata", "{}"))
                )
                
        except Exception as e:
            logger.error(f"Failed to get category {category_id}: {e}")
            raise StorageError(f"Failed to get category: {e}")
    
    async def get_category_by_name(self, name: str, domain: str = None) -> Optional[Category]:
        """Retrieve a category by name within optional domain."""
        if not self.backend.driver:
            raise ConnectionError("Database not initialized")
        
        try:
            async with self.backend.session() as session:
                if domain:
                    query = """
                    MATCH (c:Category {name: $name})
                    WHERE c.metadata CONTAINS $domain
                    RETURN c
                    LIMIT 1
                    """
                    params = {"name": name, "domain": f'"domain":"{domain}"'}
                else:
                    query = """
                    MATCH (c:Category {name: $name})
                    RETURN c
                    LIMIT 1
                    """
                    params = {"name": name}
                
                result = await session.run(query, params)
                record = await result.single()
                
                if not record:
                    return None
                
                node = record["c"]
                
                return Category(
                    category_id=node["category_id"],
                    name=node["name"],
                    level=node["level"],
                    path=node["path"],
                    labels=json.loads(node.get("labels", "{}")),
                    description=node.get("description"),
                    status=node["status"],
                    created_at=datetime.fromisoformat(node["created_at"]),
                    updated_at=datetime.fromisoformat(node["updated_at"]),
                    metadata=json.loads(node.get("metadata", "{}"))
                )
                
        except Exception as e:
            logger.error(f"Failed to get category by name {name}: {e}")
            raise StorageError(f"Failed to get category by name: {e}")
    
    async def list_categories(
        self,
        domain: Optional[str] = None,
        parent_id: Optional[str] = None,
        level: Optional[int] = None
    ) -> List[Category]:
        """List categories with optional filtering."""
        if not self.backend.driver:
            raise ConnectionError("Database not initialized")
        
        try:
            async with self.backend.session() as session:
                conditions = []
                params = {}
                
                if domain:
                    conditions.append("c.metadata CONTAINS $domain")
                    params["domain"] = f'"domain":"{domain}"'
                
                if parent_id:
                    conditions.append("(c)<-[:HAS_CHILD]-(:Category {category_id: $parent_id})")
                    params["parent_id"] = parent_id
                
                if level is not None:
                    conditions.append("c.level = $level")
                    params["level"] = level
                
                where_clause = " AND ".join(conditions) if conditions else "TRUE"
                
                query = f"""
                MATCH (c:Category)
                WHERE {where_clause}
                RETURN c
                ORDER BY c.level, c.name
                """
                
                result = await session.run(query, params)
                categories = []
                
                async for record in result:
                    node = record["c"]
                    category = Category(
                        category_id=node["category_id"],
                        name=node["name"],
                        level=node["level"],
                        path=node["path"],
                        labels=json.loads(node.get("labels", "{}")),
                        description=node.get("description"),
                        status=node["status"],
                        created_at=datetime.fromisoformat(node["created_at"]),
                        updated_at=datetime.fromisoformat(node["updated_at"]),
                        metadata=json.loads(node.get("metadata", "{}"))
                    )
                    categories.append(category)
                
                return categories
                
        except Exception as e:
            logger.error(f"Failed to list categories: {e}")
            raise StorageError(f"Failed to list categories: {e}")
    
    async def store_attribute(self, attribute: Attribute) -> str:
        """Store an attribute and its controlled values."""
        if not self.backend.driver:
            raise ConnectionError("Database not initialized")
        
        try:
            async with self.backend.session() as session:
                async with session.begin_transaction() as tx:
                    # Create attribute node
                    create_attr_query = """
                    MERGE (a:Attribute {attribute_id: $attribute_id})
                    SET a += $properties
                    RETURN a.attribute_id as id
                    """
                    
                    properties = {
                        "name": attribute.name,
                        "display_name": attribute.display_name,
                        "data_type": attribute.data_type,
                        "labels": json.dumps(attribute.labels) if attribute.labels else "{}",
                        "description": attribute.description,
                        "required": attribute.required,
                        "status": attribute.status,
                        "validation_rules": json.dumps(attribute.validation_rules) if attribute.validation_rules else "{}",
                        "created_at": attribute.created_at.isoformat(),
                        "updated_at": attribute.updated_at.isoformat(),
                        "metadata": json.dumps(attribute.metadata) if attribute.metadata else "{}"
                    }
                    
                    await tx.run(create_attr_query, {
                        "attribute_id": attribute.attribute_id,
                        "properties": properties
                    })
                    
                    # Create value nodes for categorical attributes
                    if attribute.data_type == "categorical" and attribute.values:
                        for value in attribute.values:
                            value_query = """
                            MERGE (v:AttributeValue {attribute_id: $attribute_id, value: $value})
                            SET v += $value_properties
                            """
                            
                            value_properties = {
                                "labels": json.dumps(value.labels) if value.labels else "{}",
                                "aliases": list(value.aliases) if value.aliases else [],
                                "description": value.description,
                                "status": value.status,
                                "metadata": json.dumps(value.metadata) if value.metadata else "{}"
                            }
                            
                            await tx.run(value_query, {
                                "attribute_id": attribute.attribute_id,
                                "value": value.value,
                                "value_properties": value_properties
                            })
                            
                            # Create relationship
                            rel_query = """
                            MATCH (a:Attribute {attribute_id: $attribute_id})
                            MATCH (v:AttributeValue {attribute_id: $attribute_id, value: $value})
                            MERGE (a)-[:HAS_VALUE]->(v)
                            """
                            
                            await tx.run(rel_query, {
                                "attribute_id": attribute.attribute_id,
                                "value": value.value
                            })
                
                return attribute.attribute_id
                
        except Exception as e:
            logger.error(f"Failed to store attribute: {e}")
            raise StorageError(f"Failed to store attribute: {e}")
    
    async def get_attributes_for_category(self, category_id: str) -> List[Attribute]:
        """Get all attributes associated with a category."""
        if not self.backend.driver:
            raise ConnectionError("Database not initialized")
        
        try:
            async with self.backend.session() as session:
                query = """
                MATCH (c:Category {category_id: $category_id})-[:HAS_ATTRIBUTE]->(a:Attribute)
                OPTIONAL MATCH (a)-[:HAS_VALUE]->(v:AttributeValue)
                RETURN a, collect(v) as values
                """
                
                result = await session.run(query, {"category_id": category_id})
                attributes = []
                
                async for record in result:
                    attr_node = record["a"]
                    value_nodes = record["values"]
                    
                    # Parse attribute values
                    values = []
                    for v_node in value_nodes:
                        if v_node:  # Skip null values
                            value = AttributeValue(
                                value=v_node["value"],
                                labels=json.loads(v_node.get("labels", "{}")),
                                aliases=set(v_node.get("aliases", [])),
                                description=v_node.get("description"),
                                status=v_node["status"],
                                metadata=json.loads(v_node.get("metadata", "{}"))
                            )
                            values.append(value)
                    
                    attribute = Attribute(
                        attribute_id=attr_node["attribute_id"],
                        name=attr_node["name"],
                        display_name=attr_node["display_name"],
                        data_type=attr_node["data_type"],
                        values=values,
                        validation_rules=json.loads(attr_node.get("validation_rules", "{}")),
                        labels=json.loads(attr_node.get("labels", "{}")),
                        description=attr_node.get("description"),
                        required=attr_node["required"],
                        status=attr_node["status"],
                        created_at=datetime.fromisoformat(attr_node["created_at"]),
                        updated_at=datetime.fromisoformat(attr_node["updated_at"]),
                        metadata=json.loads(attr_node.get("metadata", "{}"))
                    )
                    attributes.append(attribute)
                
                return attributes
                
        except Exception as e:
            logger.error(f"Failed to get attributes for category {category_id}: {e}")
            raise StorageError(f"Failed to get attributes: {e}")
    
    async def store_semantic_candidate(self, candidate: SemanticCandidate) -> str:
        """Store a semantic candidate and its relationships."""
        if not self.backend.driver:
            raise ConnectionError("Database not initialized")
        
        try:
            async with self.backend.session() as session:
                async with session.begin_transaction() as tx:
                    # Create candidate node
                    create_query = """
                    MERGE (s:SemanticCandidate {candidate_id: $candidate_id})
                    SET s += $properties
                    RETURN s.candidate_id as id
                    """
                    
                    properties = {
                        "term": candidate.term,
                        "normalized_term": candidate.normalized_term,
                        "frequency": candidate.frequency,
                        "contexts": candidate.contexts,
                        "cluster_id": candidate.cluster_id,
                        "score": candidate.score,
                        "language": candidate.language,
                        "status": candidate.status,
                        "created_at": candidate.created_at.isoformat(),
                        "updated_at": candidate.updated_at.isoformat(),
                        "metadata": json.dumps(candidate.metadata) if candidate.metadata else "{}"
                    }
                    
                    await tx.run(create_query, {
                        "candidate_id": candidate.candidate_id,
                        "properties": properties
                    })
                    
                    # Create semantic relationships
                    for relation in candidate.relations:
                        rel_query = """
                        MATCH (from_term:SemanticCandidate {normalized_term: $from_term})
                        MATCH (to_term:SemanticCandidate {normalized_term: $to_term})
                        MERGE (from_term)-[r:SEMANTIC_RELATION {relation_type: $relation_type}]->(to_term)
                        SET r.strength = $strength,
                            r.evidence_count = $evidence_count,
                            r.context = $context,
                            r.metadata = $metadata
                        """
                        
                        await tx.run(rel_query, {
                            "from_term": relation.from_term,
                            "to_term": relation.to_term,
                            "relation_type": relation.relation_type,
                            "strength": relation.strength,
                            "evidence_count": relation.evidence_count,
                            "context": relation.context,
                            "metadata": json.dumps(relation.metadata) if relation.metadata else "{}"
                        })
                
                return candidate.candidate_id
                
        except Exception as e:
            logger.error(f"Failed to store semantic candidate: {e}")
            raise StorageError(f"Failed to store semantic candidate: {e}")
    
    async def search_semantic_candidates(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10
    ) -> List[SemanticCandidate]:
        """Search semantic candidates by text similarity."""
        if not self.backend.driver:
            raise ConnectionError("Database not initialized")
        
        try:
            async with self.backend.session() as session:
                # Simple text matching (can be enhanced with full-text search)
                search_query = """
                MATCH (s:SemanticCandidate)
                WHERE s.normalized_term CONTAINS toLower($query) 
                   OR s.term CONTAINS toLower($query)
                """
                
                params = {"query": query.lower()}
                
                if domain:
                    search_query += " AND s.metadata CONTAINS $domain"
                    params["domain"] = f'"domain":"{domain}"'
                
                search_query += """
                RETURN s
                ORDER BY s.score DESC, s.frequency DESC
                LIMIT $limit
                """
                params["limit"] = limit
                
                result = await session.run(search_query, params)
                candidates = []
                
                async for record in result:
                    node = record["s"]
                    
                    candidate = SemanticCandidate(
                        candidate_id=node["candidate_id"],
                        term=node["term"],
                        normalized_term=node["normalized_term"],
                        frequency=node["frequency"],
                        contexts=node.get("contexts", []),
                        cluster_id=node.get("cluster_id"),
                        score=node["score"],
                        language=node["language"],
                        status=node["status"],
                        created_at=datetime.fromisoformat(node["created_at"]),
                        updated_at=datetime.fromisoformat(node["updated_at"]),
                        metadata=json.loads(node.get("metadata", "{}"))
                    )
                    candidates.append(candidate)
                
                return candidates
                
        except Exception as e:
            logger.error(f"Failed to search semantic candidates: {e}")
            raise StorageError(f"Failed to search semantic candidates: {e}")
    
    async def get_category_tree(self, root_category_id: str = None, max_depth: int = 5) -> Dict[str, Any]:
        """Get complete category tree structure."""
        if not self.backend.driver:
            raise ConnectionError("Database not initialized")
        
        try:
            async with self.backend.session() as session:
                if root_category_id:
                    query = f"""
                    MATCH path = (root:Category {{category_id: $root_id}})-[:HAS_CHILD*0..{max_depth}]->(child:Category)
                    RETURN root, child, length(path) as depth
                    ORDER BY depth, child.name
                    """
                    params = {"root_id": root_category_id}
                else:
                    query = f"""
                    MATCH path = (root:Category {{level: 0}})-[:HAS_CHILD*0..{max_depth}]->(child:Category)
                    RETURN root, child, length(path) as depth
                    ORDER BY root.name, depth, child.name
                    """
                    params = {}
                
                result = await session.run(query, params)
                tree = {}
                
                async for record in result:
                    root_node = record["root"]
                    child_node = record["child"]
                    depth = record["depth"]
                    
                    root_id = root_node["category_id"]
                    if root_id not in tree:
                        tree[root_id] = {
                            "category": self._node_to_category(root_node),
                            "children": {}
                        }
                    
                    if depth > 0:
                        # Add child to tree structure
                        current = tree[root_id]
                        for _ in range(depth - 1):
                            # Navigate to the right place in tree
                            pass
                        
                        child_id = child_node["category_id"]
                        current["children"][child_id] = {
                            "category": self._node_to_category(child_node),
                            "children": {}
                        }
                
                return tree
                
        except Exception as e:
            logger.error(f"Failed to get category tree: {e}")
            raise StorageError(f"Failed to get category tree: {e}")
    
    def _node_to_category(self, node) -> Category:
        """Convert Neo4j node to Category model."""
        return Category(
            category_id=node["category_id"],
            name=node["name"],
            level=node["level"],
            path=node["path"],
            labels=json.loads(node.get("labels", "{}")),
            description=node.get("description"),
            status=node["status"],
            created_at=datetime.fromisoformat(node["created_at"]),
            updated_at=datetime.fromisoformat(node["updated_at"]),
            metadata=json.loads(node.get("metadata", "{}"))
        )
