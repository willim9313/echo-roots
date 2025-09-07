"""Taxonomy Navigation and Tree Operations.

This module provides utilities for navigating and querying the taxonomy hierarchy,
including tree traversal, search, and structural analysis.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from echo_roots.models.taxonomy import Category
from echo_roots.storage.interfaces import TaxonomyRepository


class TraversalOrder(Enum):
    """Tree traversal order options."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST_PRE = "depth_first_pre"
    DEPTH_FIRST_POST = "depth_first_post"


@dataclass
class TreeNode:
    """Tree representation of a category with navigation utilities."""
    
    category: Category
    children: List['TreeNode']
    parent: Optional['TreeNode'] = None
    
    @property
    def is_root(self) -> bool:
        """Check if this is a root node."""
        return self.parent is None
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    @property
    def depth(self) -> int:
        """Get the depth of this node in the tree."""
        return self.category.level
    
    @property
    def subtree_size(self) -> int:
        """Get the total number of nodes in this subtree."""
        return 1 + sum(child.subtree_size for child in self.children)
    
    def find_child(self, name: str) -> Optional['TreeNode']:
        """Find a direct child by name."""
        for child in self.children:
            if child.category.name == name:
                return child
        return None
    
    def get_path_to_root(self) -> List['TreeNode']:
        """Get the path from this node to the root."""
        path = []
        current = self
        while current:
            path.insert(0, current)
            current = current.parent
        return path
    
    def get_common_ancestor(self, other: 'TreeNode') -> Optional['TreeNode']:
        """Find the lowest common ancestor with another node."""
        self_path = self.get_path_to_root()
        other_path = other.get_path_to_root()
        
        common_ancestor = None
        for self_node, other_node in zip(self_path, other_path):
            if self_node.category.category_id == other_node.category.category_id:
                common_ancestor = self_node
            else:
                break
        
        return common_ancestor


class TaxonomyNavigator:
    """Navigation utilities for taxonomy hierarchies.
    
    Provides tree-based operations, search capabilities, and structural analysis
    for taxonomy hierarchies.
    """
    
    def __init__(self, taxonomy_repo: TaxonomyRepository):
        """Initialize the navigator.
        
        Args:
            taxonomy_repo: Repository for taxonomy operations
        """
        self.taxonomy_repo = taxonomy_repo
        self._tree_cache: Dict[str, TreeNode] = {}
        self._domain_roots: Dict[str, List[TreeNode]] = {}
    
    async def build_tree(self, domain: Optional[str] = None, root_id: Optional[str] = None) -> List[TreeNode]:
        """Build a tree structure from the taxonomy hierarchy.
        
        Args:
            domain: Optional domain to filter by
            root_id: Optional root category ID to build subtree from
            
        Returns:
            List of root TreeNode objects
        """
        cache_key = f"{domain}:{root_id}"
        if cache_key in self._domain_roots:
            return self._domain_roots[cache_key]
        
        # Get all categories
        categories = await self.taxonomy_repo.list_categories(domain=domain)
        
        # Filter to subtree if root_id specified
        if root_id:
            root_category = await self.taxonomy_repo.get_category(root_id)
            if not root_category:
                return []
            
            # Include root and all descendants
            subtree_categories = [root_category]
            for cat in categories:
                if cat.path and len(cat.path) > len(root_category.path):
                    # Check if this category is a descendant
                    if cat.path[:len(root_category.path)] == root_category.path:
                        subtree_categories.append(cat)
            
            categories = subtree_categories
        
        # Create TreeNode objects
        nodes = {}
        for category in categories:
            node = TreeNode(category=category, children=[])
            nodes[category.category_id] = node
            self._tree_cache[category.category_id] = node
        
        # Build parent-child relationships
        root_nodes = []
        for category in categories:
            node = nodes[category.category_id]
            
            if category.parent_id and category.parent_id in nodes:
                parent_node = nodes[category.parent_id]
                parent_node.children.append(node)
                node.parent = parent_node
            else:
                # This is a root node (or orphaned)
                root_nodes.append(node)
        
        # Sort children by name for consistent ordering
        for node in nodes.values():
            node.children.sort(key=lambda x: x.category.name)
        
        root_nodes.sort(key=lambda x: x.category.name)
        
        # Cache results
        self._domain_roots[cache_key] = root_nodes
        
        return root_nodes
    
    async def find_category_by_path(self, path: List[str], domain: Optional[str] = None) -> Optional[Category]:
        """Find a category by its hierarchical path.
        
        Args:
            path: List of category names from root to target
            domain: Optional domain to search in
            
        Returns:
            Category if found, None otherwise
        """
        if not path:
            return None
        
        # Build tree for navigation
        roots = await self.build_tree(domain=domain)
        
        # Navigate down the path
        current_nodes = roots
        for path_component in path:
            found_node = None
            for node in current_nodes:
                if node.category.name == path_component:
                    found_node = node
                    break
            
            if not found_node:
                return None
            
            if path_component == path[-1]:
                # Found the target
                return found_node.category
            
            # Continue to children
            current_nodes = found_node.children
        
        return None
    
    async def search_categories(
        self,
        query: str,
        domain: Optional[str] = None,
        include_descriptions: bool = True,
        max_results: int = 50
    ) -> List[Tuple[Category, float]]:
        """Search for categories by name and description.
        
        Args:
            query: Search query string
            domain: Optional domain to search in
            include_descriptions: Whether to search in descriptions
            max_results: Maximum number of results to return
            
        Returns:
            List of (Category, relevance_score) tuples sorted by relevance
        """
        categories = await self.taxonomy_repo.list_categories(domain=domain)
        query_lower = query.lower().strip()
        
        if not query_lower:
            return [(cat, 1.0) for cat in categories[:max_results]]
        
        results = []
        
        for category in categories:
            score = 0.0
            
            # Exact name match gets highest score
            if category.name.lower() == query_lower:
                score = 1.0
            # Name starts with query
            elif category.name.lower().startswith(query_lower):
                score = 0.8
            # Name contains query
            elif query_lower in category.name.lower():
                score = 0.6
            # Check multilingual labels
            else:
                for label in category.labels.values():
                    if query_lower in label.lower():
                        score = max(score, 0.5)
                        break
            
            # Check description if enabled
            if include_descriptions and category.description:
                if query_lower in category.description.lower():
                    score = max(score, 0.4)
            
            # Check path components
            for path_component in category.path:
                if query_lower in path_component.lower():
                    score = max(score, 0.3)
                    break
            
            if score > 0:
                results.append((category, score))
        
        # Sort by score descending, then by name
        results.sort(key=lambda x: (-x[1], x[0].name))
        
        return results[:max_results]
    
    async def get_category_breadcrumbs(self, category_id: str) -> List[Category]:
        """Get breadcrumb trail for a category.
        
        Args:
            category_id: Category to get breadcrumbs for
            
        Returns:
            List of categories from root to target
        """
        category = await self.taxonomy_repo.get_category(category_id)
        if not category:
            return []
        
        breadcrumbs = []
        current_category = category
        
        while current_category:
            breadcrumbs.insert(0, current_category)
            
            if current_category.parent_id:
                current_category = await self.taxonomy_repo.get_category(current_category.parent_id)
            else:
                break
        
        return breadcrumbs
    
    async def traverse_tree(
        self,
        order: TraversalOrder = TraversalOrder.BREADTH_FIRST,
        domain: Optional[str] = None,
        root_id: Optional[str] = None
    ) -> List[TreeNode]:
        """Traverse the taxonomy tree in specified order.
        
        Args:
            order: Traversal order (breadth-first, depth-first pre/post)
            domain: Optional domain to traverse
            root_id: Optional root to start traversal from
            
        Returns:
            List of TreeNode objects in traversal order
        """
        roots = await self.build_tree(domain=domain, root_id=root_id)
        
        if not roots:
            return []
        
        result = []
        
        if order == TraversalOrder.BREADTH_FIRST:
            queue = roots[:]
            while queue:
                node = queue.pop(0)
                result.append(node)
                queue.extend(node.children)
        
        elif order == TraversalOrder.DEPTH_FIRST_PRE:
            def dfs_pre(node: TreeNode):
                result.append(node)
                for child in node.children:
                    dfs_pre(child)
            
            for root in roots:
                dfs_pre(root)
        
        elif order == TraversalOrder.DEPTH_FIRST_POST:
            def dfs_post(node: TreeNode):
                for child in node.children:
                    dfs_post(child)
                result.append(node)
            
            for root in roots:
                dfs_post(root)
        
        return result
    
    async def get_tree_statistics(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed tree structure statistics.
        
        Args:
            domain: Optional domain to analyze
            
        Returns:
            Dictionary with tree structure metrics
        """
        roots = await self.build_tree(domain=domain)
        
        if not roots:
            return {
                "total_nodes": 0,
                "total_roots": 0,
                "max_depth": 0,
                "avg_branching_factor": 0.0,
                "leaf_nodes": 0,
                "height_distribution": {}
            }
        
        all_nodes = await self.traverse_tree(TraversalOrder.BREADTH_FIRST, domain)
        
        total_nodes = len(all_nodes)
        total_roots = len(roots)
        
        # Calculate max depth
        max_depth = max(node.depth for node in all_nodes)
        
        # Calculate branching factor
        total_children = sum(len(node.children) for node in all_nodes)
        internal_nodes = sum(1 for node in all_nodes if node.children)
        avg_branching_factor = total_children / internal_nodes if internal_nodes > 0 else 0.0
        
        # Count leaf nodes
        leaf_nodes = sum(1 for node in all_nodes if node.is_leaf)
        
        # Height distribution
        height_distribution = {}
        for node in all_nodes:
            depth = node.depth
            height_distribution[depth] = height_distribution.get(depth, 0) + 1
        
        return {
            "total_nodes": total_nodes,
            "total_roots": total_roots,
            "max_depth": max_depth,
            "avg_branching_factor": round(avg_branching_factor, 2),
            "leaf_nodes": leaf_nodes,
            "internal_nodes": internal_nodes,
            "height_distribution": height_distribution,
            "balance_ratio": leaf_nodes / total_nodes if total_nodes > 0 else 0.0
        }
    
    async def find_similar_categories(
        self,
        category_id: str,
        similarity_threshold: float = 0.5,
        max_results: int = 10
    ) -> List[Tuple[Category, float]]:
        """Find categories similar to the given category.
        
        Args:
            category_id: Category to find similar ones for
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            List of (Category, similarity_score) tuples
        """
        target_category = await self.taxonomy_repo.get_category(category_id)
        if not target_category:
            return []
        
        # Get all categories in same domain (if applicable)
        all_categories = await self.taxonomy_repo.list_categories()
        
        similar_categories = []
        
        for category in all_categories:
            if category.category_id == category_id:
                continue
            
            similarity = self._calculate_category_similarity(target_category, category)
            
            if similarity >= similarity_threshold:
                similar_categories.append((category, similarity))
        
        # Sort by similarity descending
        similar_categories.sort(key=lambda x: -x[1])
        
        return similar_categories[:max_results]
    
    def _calculate_category_similarity(self, cat1: Category, cat2: Category) -> float:
        """Calculate similarity score between two categories.
        
        Args:
            cat1: First category
            cat2: Second category
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        score = 0.0
        
        # Name similarity (simple string matching)
        name1_lower = cat1.name.lower()
        name2_lower = cat2.name.lower()
        
        if name1_lower == name2_lower:
            score += 0.4
        elif name1_lower in name2_lower or name2_lower in name1_lower:
            score += 0.2
        
        # Level similarity (prefer same level)
        if cat1.level == cat2.level:
            score += 0.2
        
        # Path similarity (common path prefix)
        common_prefix_length = 0
        for p1, p2 in zip(cat1.path, cat2.path):
            if p1.lower() == p2.lower():
                common_prefix_length += 1
            else:
                break
        
        path_similarity = common_prefix_length / max(len(cat1.path), len(cat2.path))
        score += path_similarity * 0.3
        
        # Status similarity
        if cat1.status == cat2.status:
            score += 0.1
        
        return min(score, 1.0)
    
    def clear_cache(self) -> None:
        """Clear navigation caches."""
        self._tree_cache.clear()
        self._domain_roots.clear()
