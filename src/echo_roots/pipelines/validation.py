"""Validation and post-processing for extraction results.

This module provides validation, normalization, and quality assurance
for extraction results from LLM processing.

Key components:
- ExtractionValidator: Validates extraction results against domain schemas
- ResultNormalizer: Normalizes and cleans extraction data
- QualityAnalyzer: Analyzes extraction quality and confidence
- PostProcessor: Orchestrates validation and normalization pipeline
"""

import re
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass
from datetime import datetime

from echo_roots.models.core import ExtractionResult, AttributeExtraction, SemanticTerm
from echo_roots.models.domain import DomainPack
from echo_roots.pipelines.extraction import ExtractionError


@dataclass
class ValidationIssue:
    """Represents a validation issue found during extraction validation."""
    
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'schema', 'format', 'quality', 'consistency'
    message: str
    field: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class QualityMetrics:
    """Quality metrics for an extraction result."""
    
    avg_attribute_confidence: Optional[float]
    avg_term_confidence: Optional[float]
    schema_compliance_score: float  # 0.0-1.0
    attribute_coverage_score: float  # 0.0-1.0
    evidence_quality_score: float  # 0.0-1.0
    total_quality_score: float  # 0.0-1.0
    issue_count: int
    warning_count: int


class ExtractionValidator:
    """Validates extraction results against domain schemas and quality standards.
    
    Performs comprehensive validation including schema compliance,
    data quality checks, and domain-specific validation rules.
    """
    
    def __init__(self, domain_pack: DomainPack):
        """Initialize validator with domain configuration.
        
        Args:
            domain_pack: Domain pack with validation rules and schema
        """
        self.domain_pack = domain_pack
        self.expected_attributes = self._get_expected_attributes()
        self.validation_rules = self._get_validation_rules()
    
    def _get_expected_attributes(self) -> Dict[str, Dict[str, Any]]:
        """Extract expected attributes from domain pack schema."""
        expected = {}
        
        if 'attributes' in self.domain_pack.output_schema:
            for attr in self.domain_pack.output_schema['attributes']:
                if isinstance(attr, dict) and 'key' in attr:
                    expected[attr['key']] = attr
        
        return expected
    
    def _get_validation_rules(self) -> List[Dict[str, Any]]:
        """Extract validation rules from domain pack."""
        if hasattr(self.domain_pack, 'validation_rules'):
            return self.domain_pack.validation_rules or []
        return []
    
    def validate(self, result: ExtractionResult) -> Tuple[bool, List[ValidationIssue]]:
        """Validate an extraction result comprehensively.
        
        Args:
            result: ExtractionResult to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Schema validation
        issues.extend(self._validate_schema_compliance(result))
        
        # Attribute validation
        issues.extend(self._validate_attributes(result.attributes))
        
        # Term validation
        issues.extend(self._validate_terms(result.terms))
        
        # Quality validation
        issues.extend(self._validate_quality(result))
        
        # Domain-specific validation
        issues.extend(self._validate_domain_rules(result))
        
        # Determine if result is valid (no errors, warnings are OK)
        has_errors = any(issue.severity == 'error' for issue in issues)
        
        return not has_errors, issues
    
    def _validate_schema_compliance(self, result: ExtractionResult) -> List[ValidationIssue]:
        """Validate that result complies with expected schema."""
        issues = []
        
        # Check for required attributes
        extracted_attrs = {attr.name for attr in result.attributes}
        
        for attr_name, attr_config in self.expected_attributes.items():
            if attr_config.get('required', False) and attr_name not in extracted_attrs:
                issues.append(ValidationIssue(
                    severity='error',
                    category='schema',
                    message=f"Required attribute '{attr_name}' is missing",
                    field=attr_name,
                    suggested_fix=f"Extract {attr_name} from the source text"
                ))
        
        # Check for unexpected attributes (warnings only)
        for attr in result.attributes:
            if attr.name not in self.expected_attributes:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='schema',
                    message=f"Unexpected attribute '{attr.name}' found",
                    field=attr.name,
                    suggested_fix="Consider adding to domain schema or removing"
                ))
        
        return issues
    
    def _validate_attributes(self, attributes: List[AttributeExtraction]) -> List[ValidationIssue]:
        """Validate individual attributes."""
        issues = []
        
        for attr in attributes:
            # Validate against expected schema
            if attr.name in self.expected_attributes:
                attr_config = self.expected_attributes[attr.name]
                
                # Type validation
                if attr_config.get('type') == 'categorical':
                    allowed_values = attr_config.get('allow_values', [])
                    if allowed_values and attr.value not in allowed_values:
                        issues.append(ValidationIssue(
                            severity='warning',
                            category='schema',
                            message=f"Value '{attr.value}' not in allowed values for '{attr.name}'",
                            field=attr.name,
                            suggested_fix=f"Use one of: {allowed_values}"
                        ))
                
                # Pattern validation
                if 'pattern' in attr_config:
                    pattern = attr_config['pattern']
                    if not re.match(pattern, attr.value):
                        issues.append(ValidationIssue(
                            severity='warning',
                            category='format',
                            message=f"Value '{attr.value}' doesn't match expected pattern for '{attr.name}'",
                            field=attr.name,
                            suggested_fix=f"Format should match: {pattern}"
                        ))
            
            # General quality checks
            if len(attr.value.strip()) == 0:
                issues.append(ValidationIssue(
                    severity='error',
                    category='quality',
                    message=f"Attribute '{attr.name}' has empty value",
                    field=attr.name,
                    suggested_fix="Provide a non-empty value"
                ))
            
            if len(attr.evidence.strip()) < 3:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='quality',
                    message=f"Evidence for '{attr.name}' is too short",
                    field=attr.name,
                    suggested_fix="Provide more detailed evidence text"
                ))
            
            # Confidence validation
            if attr.confidence is not None:
                if attr.confidence < 0.0 or attr.confidence > 1.0:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='format',
                        message=f"Invalid confidence score {attr.confidence} for '{attr.name}'",
                        field=attr.name,
                        suggested_fix="Confidence must be between 0.0 and 1.0"
                    ))
                elif attr.confidence < 0.3:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='quality',
                        message=f"Low confidence score {attr.confidence} for '{attr.name}'",
                        field=attr.name,
                        suggested_fix="Consider reviewing extraction or providing more context"
                    ))
        
        return issues
    
    def _validate_terms(self, terms: List[SemanticTerm]) -> List[ValidationIssue]:
        """Validate semantic terms."""
        issues = []
        
        for term in terms:
            # Basic quality checks
            if len(term.term.strip()) == 0:
                issues.append(ValidationIssue(
                    severity='error',
                    category='quality',
                    message="Term has empty text",
                    suggested_fix="Provide non-empty term text"
                ))
            
            if len(term.context.strip()) < 5:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='quality',
                    message=f"Context for term '{term.term}' is too short",
                    suggested_fix="Provide more descriptive context"
                ))
            
            # Confidence validation
            if term.confidence < 0.0 or term.confidence > 1.0:
                issues.append(ValidationIssue(
                    severity='error',
                    category='format',
                    message=f"Invalid confidence score {term.confidence} for term '{term.term}'",
                    suggested_fix="Confidence must be between 0.0 and 1.0"
                ))
            elif term.confidence < 0.3:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='quality',
                    message=f"Low confidence score {term.confidence} for term '{term.term}'",
                    suggested_fix="Consider reviewing term extraction"
                ))
        
        return issues
    
    def _validate_quality(self, result: ExtractionResult) -> List[ValidationIssue]:
        """Validate overall extraction quality."""
        issues = []
        
        # Check for empty results
        if not result.attributes and not result.terms:
            issues.append(ValidationIssue(
                severity='warning',
                category='quality',
                message="No attributes or terms extracted",
                suggested_fix="Review source text for extractable information"
            ))
        
        # Check for duplicate attributes
        attr_names = [attr.name for attr in result.attributes]
        duplicates = set([name for name in attr_names if attr_names.count(name) > 1])
        for dup in duplicates:
            issues.append(ValidationIssue(
                severity='error',
                category='consistency',
                message=f"Duplicate attribute '{dup}' found",
                field=dup,
                suggested_fix="Combine or remove duplicate attributes"
            ))
        
        # Check for duplicate terms
        term_texts = [term.term.lower() for term in result.terms]
        dup_terms = set([term for term in term_texts if term_texts.count(term) > 1])
        for dup in dup_terms:
            issues.append(ValidationIssue(
                severity='warning',
                category='consistency',
                message=f"Duplicate term '{dup}' found",
                suggested_fix="Consider combining duplicate terms"
            ))
        
        return issues
    
    def _validate_domain_rules(self, result: ExtractionResult) -> List[ValidationIssue]:
        """Validate against domain-specific rules."""
        issues = []
        
        for rule in self.validation_rules:
            rule_type = rule.get('type')
            
            if rule_type == 'required_fields':
                required_fields = rule.get('fields', [])
                extracted_fields = {attr.name for attr in result.attributes}
                
                for field in required_fields:
                    if field not in extracted_fields:
                        issues.append(ValidationIssue(
                            severity='error',
                            category='schema',
                            message=f"Domain rule violation: required field '{field}' missing",
                            field=field,
                            suggested_fix=f"Extract {field} as required by domain rules"
                        ))
            
            elif rule_type == 'min_attributes':
                min_count = rule.get('count', 1)
                if len(result.attributes) < min_count:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='quality',
                        message=f"Domain rule: minimum {min_count} attributes required, got {len(result.attributes)}",
                        suggested_fix=f"Extract at least {min_count} attributes"
                    ))
            
            elif rule_type == 'min_terms':
                min_count = rule.get('count', 1)
                if len(result.terms) < min_count:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='quality',
                        message=f"Domain rule: minimum {min_count} terms required, got {len(result.terms)}",
                        suggested_fix=f"Extract at least {min_count} semantic terms"
                    ))
        
        return issues


class ResultNormalizer:
    """Normalizes and cleans extraction results.
    
    Applies normalization rules to ensure consistent formatting
    and clean up common extraction issues.
    """
    
    def __init__(self, domain_pack: DomainPack):
        """Initialize normalizer with domain configuration.
        
        Args:
            domain_pack: Domain pack with normalization rules
        """
        self.domain_pack = domain_pack
    
    def normalize(self, result: ExtractionResult) -> ExtractionResult:
        """Normalize an extraction result.
        
        Args:
            result: ExtractionResult to normalize
            
        Returns:
            Normalized ExtractionResult
        """
        # Normalize attributes
        normalized_attributes = [
            self._normalize_attribute(attr) for attr in result.attributes
        ]
        
        # Normalize terms
        normalized_terms = [
            self._normalize_term(term) for term in result.terms
        ]
        
        # Create new result with normalized data
        return ExtractionResult(
            item_id=result.item_id,
            attributes=normalized_attributes,
            terms=normalized_terms,
            metadata=result.metadata
        )
    
    def _normalize_attribute(self, attr: AttributeExtraction) -> AttributeExtraction:
        """Normalize a single attribute."""
        # Normalize attribute name (lowercase, underscore)
        normalized_name = self._normalize_attribute_name(attr.name)
        
        # Normalize value based on attribute type
        normalized_value = self._normalize_attribute_value(normalized_name, attr.value)
        
        # Clean evidence text
        normalized_evidence = self._clean_text(attr.evidence)
        
        return AttributeExtraction(
            name=normalized_name,
            value=normalized_value,
            evidence=normalized_evidence,
            confidence=attr.confidence
        )
    
    def _normalize_term(self, term: SemanticTerm) -> SemanticTerm:
        """Normalize a semantic term."""
        return SemanticTerm(
            term=self._clean_text(term.term),
            context=self._clean_text(term.context),
            confidence=term.confidence,
            frequency=term.frequency
        )
    
    def _normalize_attribute_name(self, name: str) -> str:
        """Normalize attribute name to standard format."""
        # Convert to lowercase and replace spaces/hyphens with underscores
        normalized = re.sub(r'[\\s-]+', '_', name.lower().strip())
        # Remove any non-alphanumeric characters except underscores
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        return normalized
    
    def _normalize_attribute_value(self, attr_name: str, value: str) -> str:
        """Normalize attribute value based on type and hints."""
        cleaned_value = self._clean_text(value)
        
        # Apply domain-specific normalization
        if attr_name in self.domain_pack.attribute_hints:
            hints = self.domain_pack.attribute_hints[attr_name]
            
            # Apply value mapping if available
            if 'value_map' in hints:
                value_map = hints['value_map']
                cleaned_value = value_map.get(cleaned_value.lower(), cleaned_value)
        
        # Apply type-specific normalization
        if 'attributes' in self.domain_pack.output_schema:
            for attr_config in self.domain_pack.output_schema['attributes']:
                if isinstance(attr_config, dict) and attr_config.get('key') == attr_name:
                    attr_type = attr_config.get('type')
                    
                    if attr_type == 'categorical':
                        # Normalize categorical values
                        allowed_values = attr_config.get('allow_values', [])
                        if allowed_values:
                            # Try to match case-insensitively
                            for allowed_val in allowed_values:
                                if cleaned_value.lower() == allowed_val.lower():
                                    cleaned_value = allowed_val
                                    break
                    
                    elif attr_type == 'numeric':
                        # Clean numeric values
                        cleaned_value = re.sub(r'[^0-9.-]', '', cleaned_value)
                    
                    elif attr_type == 'boolean':
                        # Normalize boolean values
                        if cleaned_value.lower() in ['true', '1', 'yes', 'y']:
                            cleaned_value = 'true'
                        elif cleaned_value.lower() in ['false', '0', 'no', 'n']:
                            cleaned_value = 'false'
                    
                    break
        
        return cleaned_value
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\\s+', ' ', text.strip())
        
        # Remove special characters that might cause issues
        cleaned = re.sub(r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f]', '', cleaned)
        
        return cleaned


class QualityAnalyzer:
    """Analyzes extraction result quality and provides metrics."""
    
    def __init__(self, domain_pack: DomainPack):
        """Initialize quality analyzer.
        
        Args:
            domain_pack: Domain pack for quality benchmarks
        """
        self.domain_pack = domain_pack
    
    def analyze(self, result: ExtractionResult, validation_issues: List[ValidationIssue]) -> QualityMetrics:
        """Analyze extraction result quality.
        
        Args:
            result: ExtractionResult to analyze
            validation_issues: List of validation issues found
            
        Returns:
            QualityMetrics with calculated scores
        """
        # Calculate confidence averages
        attr_confidences = [attr.confidence for attr in result.attributes if attr.confidence is not None]
        avg_attr_confidence = sum(attr_confidences) / len(attr_confidences) if attr_confidences else None
        
        term_confidences = [term.confidence for term in result.terms]
        avg_term_confidence = sum(term_confidences) / len(term_confidences) if term_confidences else None
        
        # Calculate schema compliance score
        schema_score = self._calculate_schema_compliance_score(result, validation_issues)
        
        # Calculate attribute coverage score
        coverage_score = self._calculate_coverage_score(result)
        
        # Calculate evidence quality score
        evidence_score = self._calculate_evidence_quality_score(result)
        
        # Calculate overall quality score
        total_score = self._calculate_total_quality_score(
            schema_score, coverage_score, evidence_score, validation_issues
        )
        
        # Count issues by severity
        error_count = sum(1 for issue in validation_issues if issue.severity == 'error')
        warning_count = sum(1 for issue in validation_issues if issue.severity == 'warning')
        
        return QualityMetrics(
            avg_attribute_confidence=avg_attr_confidence,
            avg_term_confidence=avg_term_confidence,
            schema_compliance_score=schema_score,
            attribute_coverage_score=coverage_score,
            evidence_quality_score=evidence_score,
            total_quality_score=total_score,
            issue_count=error_count,
            warning_count=warning_count
        )
    
    def _calculate_schema_compliance_score(self, result: ExtractionResult, issues: List[ValidationIssue]) -> float:
        """Calculate how well the result complies with the schema."""
        schema_errors = [issue for issue in issues if issue.category == 'schema' and issue.severity == 'error']
        
        if not schema_errors:
            return 1.0
        
        # Penalize based on number of schema errors
        penalty = min(len(schema_errors) * 0.2, 0.8)
        return max(1.0 - penalty, 0.2)
    
    def _calculate_coverage_score(self, result: ExtractionResult) -> float:
        """Calculate attribute coverage score."""
        expected_attrs = set()
        if 'attributes' in self.domain_pack.output_schema:
            for attr in self.domain_pack.output_schema['attributes']:
                if isinstance(attr, dict) and 'key' in attr:
                    expected_attrs.add(attr['key'])
        
        if not expected_attrs:
            return 1.0  # No expectations, perfect coverage
        
        extracted_attrs = {attr.name for attr in result.attributes}
        coverage = len(expected_attrs.intersection(extracted_attrs)) / len(expected_attrs)
        
        return coverage
    
    def _calculate_evidence_quality_score(self, result: ExtractionResult) -> float:
        """Calculate evidence quality score based on evidence length and content."""
        if not result.attributes:
            return 1.0  # No attributes to evaluate
        
        scores = []
        for attr in result.attributes:
            evidence_len = len(attr.evidence.strip())
            if evidence_len == 0:
                scores.append(0.0)
            elif evidence_len < 10:
                scores.append(0.3)
            elif evidence_len < 20:
                scores.append(0.6)
            else:
                scores.append(1.0)
        
        return sum(scores) / len(scores)
    
    def _calculate_total_quality_score(self, schema_score: float, coverage_score: float, 
                                     evidence_score: float, issues: List[ValidationIssue]) -> float:
        """Calculate overall quality score."""
        # Weighted average of individual scores
        base_score = (schema_score * 0.4 + coverage_score * 0.3 + evidence_score * 0.3)
        
        # Apply penalties for issues
        error_penalty = sum(0.1 for issue in issues if issue.severity == 'error')
        warning_penalty = sum(0.05 for issue in issues if issue.severity == 'warning')
        
        total_penalty = min(error_penalty + warning_penalty, 0.5)
        final_score = max(base_score - total_penalty, 0.0)
        
        return final_score


class PostProcessor:
    """Orchestrates validation and normalization of extraction results.
    
    Provides a unified interface for post-processing extraction results
    with validation, normalization, and quality analysis.
    """
    
    def __init__(self, domain_pack: DomainPack):
        """Initialize post-processor.
        
        Args:
            domain_pack: Domain pack for validation and normalization
        """
        self.domain_pack = domain_pack
        self.validator = ExtractionValidator(domain_pack)
        self.normalizer = ResultNormalizer(domain_pack)
        self.quality_analyzer = QualityAnalyzer(domain_pack)
    
    def process(self, result: ExtractionResult, 
               normalize: bool = True,
               validate: bool = True) -> Tuple[ExtractionResult, List[ValidationIssue], Optional[QualityMetrics]]:
        """Process an extraction result with validation and normalization.
        
        Args:
            result: ExtractionResult to process
            normalize: Whether to apply normalization
            validate: Whether to perform validation
            
        Returns:
            Tuple of (processed_result, validation_issues, quality_metrics)
        """
        processed_result = result
        validation_issues = []
        quality_metrics = None
        
        # Normalize first if requested
        if normalize:
            processed_result = self.normalizer.normalize(processed_result)
        
        # Validate if requested
        if validate:
            is_valid, validation_issues = self.validator.validate(processed_result)
            quality_metrics = self.quality_analyzer.analyze(processed_result, validation_issues)
        
        return processed_result, validation_issues, quality_metrics
    
    def process_batch(self, results: List[ExtractionResult],
                     normalize: bool = True,
                     validate: bool = True) -> List[Tuple[ExtractionResult, List[ValidationIssue], Optional[QualityMetrics]]]:
        """Process multiple extraction results.
        
        Args:
            results: List of ExtractionResults to process
            normalize: Whether to apply normalization
            validate: Whether to perform validation
            
        Returns:
            List of tuples (processed_result, validation_issues, quality_metrics)
        """
        return [
            self.process(result, normalize=normalize, validate=validate)
            for result in results
        ]
