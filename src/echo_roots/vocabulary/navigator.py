# T7 Controlled Vocabulary Navigation and Utilities

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, DefaultDict
from collections import defaultdict, deque
import logging
import re
from datetime import datetime, timedelta, UTC

from .manager import (
    VocabularyTerm, VocabularyType, VocabularyRepository, 
    VocabularyNormalizer, MappingConfidence
)

logger = logging.getLogger(__name__)


@dataclass
class VocabularyHierarchy:
    """Represents a vocabulary hierarchy tree."""
    root_terms: List[VocabularyTerm] = field(default_factory=list)
    term_children: Dict[str, List[VocabularyTerm]] = field(default_factory=dict)
    term_parents: Dict[str, VocabularyTerm] = field(default_factory=dict)
    depth_map: Dict[str, int] = field(default_factory=dict)
    vocabulary_type: Optional[VocabularyType] = None
    category_id: Optional[str] = None


@dataclass
class VocabularyCluster:
    """A cluster of related vocabulary terms."""
    cluster_id: str
    center_term: VocabularyTerm
    related_terms: List[Tuple[VocabularyTerm, float]] = field(default_factory=list)  # (term, similarity)
    cluster_type: str = "semantic"  # semantic, syntactic, usage
    strength: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VocabularyRecommendation:
    """Recommendation for vocabulary improvement."""
    recommendation_id: str
    recommendation_type: str  # missing_term, merge_terms, split_term, update_hierarchy
    priority: str = "medium"  # low, medium, high, critical
    description: str = ""
    affected_terms: List[str] = field(default_factory=list)
    suggested_action: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class VocabularyNavigator:
    """Navigation utilities for vocabulary hierarchies and relationships."""
    
    def __init__(self, repository: VocabularyRepository):
        self.repository = repository
        self.normalizer = VocabularyNormalizer()
        self._hierarchy_cache: Dict[str, VocabularyHierarchy] = {}
        self._relationship_cache: Dict[str, List[Tuple[str, str, float]]] = {}
    
    async def build_hierarchy(
        self,
        vocabulary_type: VocabularyType,
        category_id: Optional[str] = None,
        force_refresh: bool = False
    ) -> VocabularyHierarchy:
        """Build vocabulary hierarchy for a given type and category."""
        cache_key = f"{vocabulary_type}:{category_id}"
        
        if not force_refresh and cache_key in self._hierarchy_cache:
            return self._hierarchy_cache[cache_key]
        
        # Get all terms of this type
        terms = await self.repository.find_terms(
            vocabulary_type=vocabulary_type,
            category_id=category_id
        )
        
        # Build hierarchy structure
        hierarchy = VocabularyHierarchy(
            vocabulary_type=vocabulary_type,
            category_id=category_id
        )
        
        # Organize terms by parent-child relationships
        terms_by_id = {term.term_id: term for term in terms}
        
        for term in terms:
            if term.parent_term_id and term.parent_term_id in terms_by_id:
                # This is a child term
                parent = terms_by_id[term.parent_term_id]
                if parent.term_id not in hierarchy.term_children:
                    hierarchy.term_children[parent.term_id] = []
                hierarchy.term_children[parent.term_id].append(term)
                hierarchy.term_parents[term.term_id] = parent
            else:
                # This is a root term
                hierarchy.root_terms.append(term)
        
        # Calculate depths
        self._calculate_depths(hierarchy)
        
        # Cache hierarchy
        self._hierarchy_cache[cache_key] = hierarchy
        
        logger.info(f"Built hierarchy for {vocabulary_type} with {len(terms)} terms")
        return hierarchy
    
    async def find_term_path(
        self,
        term_id: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str] = None
    ) -> List[VocabularyTerm]:
        """Find the path from root to a specific term."""
        hierarchy = await self.build_hierarchy(vocabulary_type, category_id)
        
        path = []
        current_term_id = term_id
        
        # Traverse up to root
        while current_term_id in hierarchy.term_parents:
            parent = hierarchy.term_parents[current_term_id]
            path.insert(0, parent)
            current_term_id = parent.term_id
        
        # Add the target term itself
        term = await self.repository.get_term(term_id)
        if term:
            path.append(term)
        
        return path
    
    async def get_term_descendants(
        self,
        term_id: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str] = None,
        max_depth: Optional[int] = None
    ) -> List[VocabularyTerm]:
        """Get all descendants of a term."""
        hierarchy = await self.build_hierarchy(vocabulary_type, category_id)
        
        descendants = []
        queue = deque([(term_id, 0)])  # (term_id, depth)
        
        while queue:
            current_id, depth = queue.popleft()
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            if current_id in hierarchy.term_children:
                for child in hierarchy.term_children[current_id]:
                    descendants.append(child)
                    queue.append((child.term_id, depth + 1))
        
        return descendants
    
    async def get_term_siblings(
        self,
        term_id: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str] = None
    ) -> List[VocabularyTerm]:
        """Get sibling terms (same parent)."""
        hierarchy = await self.build_hierarchy(vocabulary_type, category_id)
        
        if term_id not in hierarchy.term_parents:
            # Root term - siblings are other root terms
            term = await self.repository.get_term(term_id)
            if term in hierarchy.root_terms:
                return [t for t in hierarchy.root_terms if t.term_id != term_id]
            return []
        
        parent = hierarchy.term_parents[term_id]
        siblings = hierarchy.term_children.get(parent.term_id, [])
        return [s for s in siblings if s.term_id != term_id]
    
    async def search_related_terms(
        self,
        term_id: str,
        relation_types: Optional[List[str]] = None,
        max_results: int = 20
    ) -> List[Tuple[VocabularyTerm, str, float]]:
        """Search for terms related to a given term."""
        term = await self.repository.get_term(term_id)
        if not term:
            return []
        
        related = []
        
        # Get hierarchy relationships
        hierarchy_related = await self._get_hierarchy_related(term)
        related.extend(hierarchy_related)
        
        # Get semantic relationships (synonyms, similar terms)
        semantic_related = await self._get_semantic_related(term)
        related.extend(semantic_related)
        
        # Get co-occurrence relationships
        cooccurrence_related = await self._get_cooccurrence_related(term)
        related.extend(cooccurrence_related)
        
        # Filter by relation types if specified
        if relation_types:
            related = [(t, r, s) for t, r, s in related if r in relation_types]
        
        # Sort by relevance and limit results
        related.sort(key=lambda x: x[2], reverse=True)
        return related[:max_results]
    
    async def cluster_vocabulary(
        self,
        vocabulary_type: VocabularyType,
        category_id: Optional[str] = None,
        clustering_method: str = "semantic"
    ) -> List[VocabularyCluster]:
        """Cluster vocabulary terms by similarity."""
        terms = await self.repository.find_terms(vocabulary_type, category_id)
        
        if clustering_method == "semantic":
            return await self._cluster_semantic(terms)
        elif clustering_method == "syntactic":
            return await self._cluster_syntactic(terms)
        elif clustering_method == "usage":
            return await self._cluster_usage(terms)
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
    
    def _calculate_depths(self, hierarchy: VocabularyHierarchy) -> None:
        """Calculate depth for each term in hierarchy."""
        # BFS to calculate depths
        queue = deque([(term.term_id, 0) for term in hierarchy.root_terms])
        
        while queue:
            term_id, depth = queue.popleft()
            hierarchy.depth_map[term_id] = depth
            
            if term_id in hierarchy.term_children:
                for child in hierarchy.term_children[term_id]:
                    queue.append((child.term_id, depth + 1))
    
    async def _get_hierarchy_related(self, term: VocabularyTerm) -> List[Tuple[VocabularyTerm, str, float]]:
        """Get hierarchy-based related terms."""
        related = []
        
        # Get parent
        if term.parent_term_id:
            parent = await self.repository.get_term(term.parent_term_id)
            if parent:
                related.append((parent, "parent", 0.9))
        
        # Get children
        children = await self.get_term_descendants(
            term.term_id, term.vocabulary_type, term.category_id, max_depth=1
        )
        for child in children:
            related.append((child, "child", 0.8))
        
        # Get siblings
        siblings = await self.get_term_siblings(
            term.term_id, term.vocabulary_type, term.category_id
        )
        for sibling in siblings:
            related.append((sibling, "sibling", 0.7))
        
        return related
    
    async def _get_semantic_related(self, term: VocabularyTerm) -> List[Tuple[VocabularyTerm, str, float]]:
        """Get semantically related terms."""
        related = []
        
        # Find terms with similar names/descriptions
        all_terms = await self.repository.find_terms(
            vocabulary_type=term.vocabulary_type,
            category_id=term.category_id
        )
        
        normalized_term = self.normalizer.normalize_term(term.term)
        
        for other_term in all_terms:
            if other_term.term_id == term.term_id:
                continue
            
            # Check aliases and synonyms
            all_variants = [other_term.term] + other_term.aliases + other_term.synonyms
            for variant in all_variants:
                normalized_variant = self.normalizer.normalize_term(variant)
                similarity = self._calculate_string_similarity(normalized_term, normalized_variant)
                
                if similarity > 0.6:
                    relation_type = "synonym" if variant in other_term.synonyms else "similar"
                    related.append((other_term, relation_type, similarity))
                    break
        
        return related
    
    async def _get_cooccurrence_related(self, term: VocabularyTerm) -> List[Tuple[VocabularyTerm, str, float]]:
        """Get co-occurrence based related terms."""
        # This would analyze usage patterns to find frequently co-occurring terms
        # For now, return empty list - would need usage data
        return []
    
    async def _cluster_semantic(self, terms: List[VocabularyTerm]) -> List[VocabularyCluster]:
        """Cluster terms semantically."""
        clusters = []
        used_terms = set()
        
        for term in terms:
            if term.term_id in used_terms:
                continue
            
            # Find similar terms
            cluster_terms = []
            normalized_term = self.normalizer.normalize_term(term.term)
            
            for other_term in terms:
                if other_term.term_id == term.term_id or other_term.term_id in used_terms:
                    continue
                
                normalized_other = self.normalizer.normalize_term(other_term.term)
                similarity = self._calculate_string_similarity(normalized_term, normalized_other)
                
                if similarity > 0.7:  # Threshold for clustering
                    cluster_terms.append((other_term, similarity))
                    used_terms.add(other_term.term_id)
            
            if cluster_terms:
                cluster = VocabularyCluster(
                    cluster_id=f"cluster_{term.term_id}",
                    center_term=term,
                    related_terms=cluster_terms,
                    cluster_type="semantic",
                    strength=sum(s for _, s in cluster_terms) / len(cluster_terms)
                )
                clusters.append(cluster)
                used_terms.add(term.term_id)
        
        return clusters
    
    async def _cluster_syntactic(self, terms: List[VocabularyTerm]) -> List[VocabularyCluster]:
        """Cluster terms by syntactic patterns."""
        # Group by patterns like prefixes, suffixes, etc.
        pattern_groups = defaultdict(list)
        
        for term in terms:
            normalized = self.normalizer.normalize_term(term.term)
            
            # Extract patterns
            if ' ' in normalized:
                # Multi-word terms
                words = normalized.split()
                if len(words) == 2:
                    pattern_groups[f"two_word_{words[0]}"].append(term)
                    pattern_groups[f"two_word_{words[1]}"].append(term)
            
            # Single word patterns
            if len(normalized) > 3:
                prefix = normalized[:3]
                suffix = normalized[-3:]
                pattern_groups[f"prefix_{prefix}"].append(term)
                pattern_groups[f"suffix_{suffix}"].append(term)
        
        clusters = []
        for pattern, pattern_terms in pattern_groups.items():
            if len(pattern_terms) > 1:
                center_term = max(pattern_terms, key=lambda t: t.usage_count)
                related_terms = [(t, 0.8) for t in pattern_terms if t != center_term]
                
                cluster = VocabularyCluster(
                    cluster_id=f"syntactic_{pattern}",
                    center_term=center_term,
                    related_terms=related_terms,
                    cluster_type="syntactic",
                    strength=len(related_terms) / 10.0  # Normalize by cluster size
                )
                clusters.append(cluster)
        
        return clusters
    
    async def _cluster_usage(self, terms: List[VocabularyTerm]) -> List[VocabularyCluster]:
        """Cluster terms by usage patterns."""
        # Group by usage frequency ranges
        usage_ranges = {
            "high_usage": [],
            "medium_usage": [],
            "low_usage": [],
            "unused": []
        }
        
        for term in terms:
            if term.usage_count > 100:
                usage_ranges["high_usage"].append(term)
            elif term.usage_count > 10:
                usage_ranges["medium_usage"].append(term)
            elif term.usage_count > 0:
                usage_ranges["low_usage"].append(term)
            else:
                usage_ranges["unused"].append(term)
        
        clusters = []
        for usage_type, usage_terms in usage_ranges.items():
            if len(usage_terms) > 1:
                center_term = max(usage_terms, key=lambda t: t.usage_count)
                related_terms = [(t, 0.6) for t in usage_terms if t != center_term]
                
                cluster = VocabularyCluster(
                    cluster_id=f"usage_{usage_type}",
                    center_term=center_term,
                    related_terms=related_terms,
                    cluster_type="usage",
                    strength=len(related_terms) / 20.0
                )
                clusters.append(cluster)
        
        return clusters
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using simple algorithm."""
        if str1 == str2:
            return 1.0
        
        # Jaccard similarity on character bigrams
        bigrams1 = set(str1[i:i+2] for i in range(len(str1)-1))
        bigrams2 = set(str2[i:i+2] for i in range(len(str2)-1))
        
        if not bigrams1 and not bigrams2:
            return 1.0
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        
        return intersection / union if union > 0 else 0.0


class VocabularyAnalyzer:
    """Analyzes vocabulary for quality, coverage, and improvement opportunities."""
    
    def __init__(self, repository: VocabularyRepository, navigator: VocabularyNavigator):
        self.repository = repository
        self.navigator = navigator
    
    async def analyze_vocabulary_quality(
        self,
        vocabulary_type: Optional[VocabularyType] = None,
        category_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze overall vocabulary quality."""
        terms = await self.repository.find_terms(vocabulary_type, category_id)
        
        quality_metrics = {
            "total_terms": len(terms),
            "completeness_score": 0.0,
            "consistency_score": 0.0,
            "coverage_score": 0.0,
            "redundancy_score": 0.0,
            "hierarchy_health": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        if not terms:
            quality_metrics["issues"].append("No vocabulary terms found")
            return quality_metrics
        
        # Analyze completeness
        completeness = await self._analyze_completeness(terms)
        quality_metrics["completeness_score"] = completeness["score"]
        quality_metrics["issues"].extend(completeness["issues"])
        
        # Analyze consistency
        consistency = await self._analyze_consistency(terms)
        quality_metrics["consistency_score"] = consistency["score"]
        quality_metrics["issues"].extend(consistency["issues"])
        
        # Analyze coverage
        coverage = await self._analyze_coverage(terms)
        quality_metrics["coverage_score"] = coverage["score"]
        quality_metrics["issues"].extend(coverage["issues"])
        
        # Analyze redundancy
        redundancy = await self._analyze_redundancy(terms)
        quality_metrics["redundancy_score"] = redundancy["score"]
        quality_metrics["issues"].extend(redundancy["issues"])
        
        # Analyze hierarchy health
        if vocabulary_type:
            hierarchy_health = await self._analyze_hierarchy_health(vocabulary_type, category_id)
            quality_metrics["hierarchy_health"] = hierarchy_health["score"]
            quality_metrics["issues"].extend(hierarchy_health["issues"])
        
        return quality_metrics
    
    async def generate_recommendations(
        self,
        vocabulary_type: Optional[VocabularyType] = None,
        category_id: Optional[str] = None
    ) -> List[VocabularyRecommendation]:
        """Generate recommendations for vocabulary improvement."""
        recommendations = []
        
        # Analyze missing terms
        missing_recommendations = await self._recommend_missing_terms(vocabulary_type, category_id)
        recommendations.extend(missing_recommendations)
        
        # Recommend merging similar terms
        merge_recommendations = await self._recommend_term_merges(vocabulary_type, category_id)
        recommendations.extend(merge_recommendations)
        
        # Recommend hierarchy improvements
        hierarchy_recommendations = await self._recommend_hierarchy_improvements(vocabulary_type, category_id)
        recommendations.extend(hierarchy_recommendations)
        
        # Recommend cleanup actions
        cleanup_recommendations = await self._recommend_cleanup_actions(vocabulary_type, category_id)
        recommendations.extend(cleanup_recommendations)
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda r: (
            {"critical": 4, "high": 3, "medium": 2, "low": 1}[r.priority],
            r.confidence
        ), reverse=True)
        
        return recommendations
    
    async def _analyze_completeness(self, terms: List[VocabularyTerm]) -> Dict[str, Any]:
        """Analyze vocabulary completeness."""
        issues = []
        
        # Check for empty descriptions
        terms_without_descriptions = [t for t in terms if not t.description]
        if terms_without_descriptions:
            issues.append(f"{len(terms_without_descriptions)} terms lack descriptions")
        
        # Check for missing labels
        terms_without_labels = [t for t in terms if not t.labels]
        if terms_without_labels:
            issues.append(f"{len(terms_without_labels)} terms lack multilingual labels")
        
        # Calculate completeness score
        total_checks = len(terms) * 2  # Description + labels
        missing_items = len(terms_without_descriptions) + len(terms_without_labels)
        score = max(0.0, 1.0 - (missing_items / total_checks))
        
        return {"score": score, "issues": issues}
    
    async def _analyze_consistency(self, terms: List[VocabularyTerm]) -> Dict[str, Any]:
        """Analyze vocabulary consistency."""
        issues = []
        normalizer = VocabularyNormalizer()
        
        # Check for case inconsistencies
        term_variants = defaultdict(list)
        for term in terms:
            normalized = normalizer.normalize_term(term.term)
            term_variants[normalized].append(term)
        
        case_inconsistencies = [variants for variants in term_variants.values() if len(variants) > 1]
        if case_inconsistencies:
            issues.append(f"{len(case_inconsistencies)} groups of terms with case inconsistencies")
        
        # Check for naming pattern inconsistencies
        pattern_violations = 0
        for term in terms:
            if re.search(r'[A-Z]{2,}', term.term):  # Multiple consecutive capitals
                pattern_violations += 1
        
        if pattern_violations:
            issues.append(f"{pattern_violations} terms violate naming patterns")
        
        # Calculate consistency score
        total_issues = len(case_inconsistencies) + pattern_violations
        score = max(0.0, 1.0 - (total_issues / len(terms)))
        
        return {"score": score, "issues": issues}
    
    async def _analyze_coverage(self, terms: List[VocabularyTerm]) -> Dict[str, Any]:
        """Analyze vocabulary coverage."""
        issues = []
        
        # Analyze usage distribution
        used_terms = [t for t in terms if t.usage_count > 0]
        unused_terms = [t for t in terms if t.usage_count == 0]
        
        if unused_terms:
            issues.append(f"{len(unused_terms)} terms are never used")
        
        # Check for coverage gaps (would need domain knowledge)
        coverage_score = len(used_terms) / len(terms) if terms else 0.0
        
        return {"score": coverage_score, "issues": issues}
    
    async def _analyze_redundancy(self, terms: List[VocabularyTerm]) -> Dict[str, Any]:
        """Analyze vocabulary redundancy."""
        issues = []
        navigator = VocabularyNavigator(self.repository)
        
        # Find potentially duplicate terms
        duplicates = 0
        for i, term1 in enumerate(terms):
            for term2 in terms[i+1:]:
                similarity = navigator._calculate_string_similarity(
                    navigator.normalizer.normalize_term(term1.term),
                    navigator.normalizer.normalize_term(term2.term)
                )
                if similarity > 0.9:  # Very similar terms
                    duplicates += 1
        
        if duplicates:
            issues.append(f"{duplicates} pairs of potentially duplicate terms")
        
        # Calculate redundancy score (lower is better)
        redundancy_rate = duplicates / len(terms) if terms else 0.0
        score = max(0.0, 1.0 - redundancy_rate)
        
        return {"score": score, "issues": issues}
    
    async def _analyze_hierarchy_health(
        self,
        vocabulary_type: VocabularyType,
        category_id: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze hierarchy health."""
        issues = []
        
        try:
            hierarchy = await self.navigator.build_hierarchy(vocabulary_type, category_id)
            
            # Check for orphaned terms (terms with non-existent parents)
            orphaned_count = 0
            for term_id, parent in hierarchy.term_parents.items():
                if parent.term_id not in [t.term_id for t in await self.repository.find_terms()]:
                    orphaned_count += 1
            
            if orphaned_count:
                issues.append(f"{orphaned_count} orphaned terms found")
            
            # Check hierarchy depth
            max_depth = max(hierarchy.depth_map.values()) if hierarchy.depth_map else 0
            if max_depth > 7:  # Deep hierarchies can be problematic
                issues.append(f"Hierarchy too deep: {max_depth} levels")
            
            # Check for unbalanced trees
            root_children_counts = [len(hierarchy.term_children.get(root.term_id, [])) for root in hierarchy.root_terms]
            if root_children_counts and max(root_children_counts) > 20:
                issues.append("Some hierarchy branches are too wide")
            
            # Calculate health score
            total_terms = len(hierarchy.depth_map)
            issue_weight = orphaned_count + (1 if max_depth > 7 else 0) + (1 if max(root_children_counts, default=0) > 20 else 0)
            score = max(0.0, 1.0 - (issue_weight / max(total_terms, 1)))
            
        except Exception as e:
            logger.error(f"Error analyzing hierarchy health: {e}")
            issues.append("Failed to analyze hierarchy structure")
            score = 0.0
        
        return {"score": score, "issues": issues}
    
    async def _recommend_missing_terms(
        self,
        vocabulary_type: Optional[VocabularyType],
        category_id: Optional[str]
    ) -> List[VocabularyRecommendation]:
        """Recommend missing vocabulary terms."""
        # This would analyze usage patterns to suggest missing terms
        # For now, return empty list - would need usage analytics
        return []
    
    async def _recommend_term_merges(
        self,
        vocabulary_type: Optional[VocabularyType],
        category_id: Optional[str]
    ) -> List[VocabularyRecommendation]:
        """Recommend merging similar terms."""
        recommendations = []
        
        if vocabulary_type:
            clusters = await self.navigator.cluster_vocabulary(vocabulary_type, category_id, "semantic")
            
            for cluster in clusters:
                if len(cluster.related_terms) > 0 and cluster.strength > 0.8:
                    recommendation = VocabularyRecommendation(
                        recommendation_id=f"merge_{cluster.cluster_id}",
                        recommendation_type="merge_terms",
                        priority="medium",
                        description=f"Consider merging similar terms in cluster '{cluster.center_term.term}'",
                        affected_terms=[cluster.center_term.term_id] + [t.term_id for t, _ in cluster.related_terms],
                        suggested_action={
                            "action": "merge_terms",
                            "keep_term": cluster.center_term.term_id,
                            "merge_terms": [t.term_id for t, _ in cluster.related_terms]
                        },
                        confidence=cluster.strength,
                        supporting_evidence=[f"Semantic similarity: {cluster.strength:.2f}"]
                    )
                    recommendations.append(recommendation)
        
        return recommendations
    
    async def _recommend_hierarchy_improvements(
        self,
        vocabulary_type: Optional[VocabularyType],
        category_id: Optional[str]
    ) -> List[VocabularyRecommendation]:
        """Recommend hierarchy structure improvements."""
        recommendations = []
        
        if vocabulary_type:
            try:
                hierarchy = await self.navigator.build_hierarchy(vocabulary_type, category_id)
                
                # Recommend creating intermediate levels for wide branches
                for root in hierarchy.root_terms:
                    children = hierarchy.term_children.get(root.term_id, [])
                    if len(children) > 15:
                        recommendation = VocabularyRecommendation(
                            recommendation_id=f"restructure_{root.term_id}",
                            recommendation_type="update_hierarchy",
                            priority="high",
                            description=f"Term '{root.term}' has too many direct children ({len(children)})",
                            affected_terms=[root.term_id],
                            suggested_action={
                                "action": "create_intermediate_levels",
                                "parent_term": root.term_id,
                                "child_count": len(children)
                            },
                            confidence=0.8,
                            supporting_evidence=[f"Direct children: {len(children)} (recommended max: 15)"]
                        )
                        recommendations.append(recommendation)
                        
            except Exception as e:
                logger.error(f"Error analyzing hierarchy for recommendations: {e}")
        
        return recommendations
    
    async def _recommend_cleanup_actions(
        self,
        vocabulary_type: Optional[VocabularyType],
        category_id: Optional[str]
    ) -> List[VocabularyRecommendation]:
        """Recommend cleanup actions."""
        recommendations = []
        
        terms = await self.repository.find_terms(vocabulary_type, category_id)
        
        # Recommend removing unused terms
        unused_terms = [t for t in terms if t.usage_count == 0 and 
                       (datetime.now(UTC) - t.created_at).days > 90]
        
        if unused_terms:
            recommendation = VocabularyRecommendation(
                recommendation_id="cleanup_unused",
                recommendation_type="cleanup",
                priority="low",
                description=f"Remove {len(unused_terms)} unused terms older than 90 days",
                affected_terms=[t.term_id for t in unused_terms],
                suggested_action={
                    "action": "remove_unused_terms",
                    "term_ids": [t.term_id for t in unused_terms]
                },
                confidence=0.9,
                supporting_evidence=[f"Terms unused for 90+ days: {len(unused_terms)}"]
            )
            recommendations.append(recommendation)
        
        return recommendations
