import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    data_quality_score: float

class DataValidator:
    """Comprehensive data validation for the pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_dataframe(self, df: pd.DataFrame, text_columns: List[str], 
                          id_column: str = None) -> ValidationResult:
        """Validate input dataframe"""
        errors = []
        warnings = []
        suggestions = []
        
        # Basic checks
        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, suggestions, 0.0)
        
        # Check required columns
        missing_columns = [col for col in text_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required text columns: {missing_columns}")
        
        if id_column and id_column not in df.columns:
            errors.append(f"Missing ID column: {id_column}")
        
        # Check for null values in text columns
        for col in text_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                null_percentage = (null_count / len(df)) * 100
                
                if null_percentage > 50:
                    errors.append(f"Column '{col}' has {null_percentage:.1f}% null values")
                elif null_percentage > 20:
                    warnings.append(f"Column '{col}' has {null_percentage:.1f}% null values")
        
        # Check text quality
        for col in text_columns:
            if col in df.columns:
                text_lengths = df[col].dropna().astype(str).str.len()
                avg_length = text_lengths.mean()
                
                if avg_length < 10:
                    warnings.append(f"Column '{col}' has very short text (avg: {avg_length:.1f} chars)")
                elif avg_length > 10000:
                    warnings.append(f"Column '{col}' has very long text (avg: {avg_length:.1f} chars)")
        
        # Check for duplicates
        if id_column and id_column in df.columns:
            duplicate_count = df[id_column].duplicated().sum()
            if duplicate_count > 0:
                errors.append(f"Found {duplicate_count} duplicate IDs")
        
        # Data quality suggestions
        if len(df) < 10:
            suggestions.append("Consider adding more data for better graph construction")
        elif len(df) > 10000:
            suggestions.append("Large dataset - consider sampling for faster processing")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df, text_columns, errors, warnings)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, suggestions, quality_score)
    
    def _calculate_quality_score(self, df: pd.DataFrame, text_columns: List[str], 
                               errors: List[str], warnings: List[str]) -> float:
        """Calculate data quality score (0-1)"""
        score = 1.0
        
        # Penalize errors heavily
        score -= len(errors) * 0.3
        
        # Penalize warnings moderately
        score -= len(warnings) * 0.1
        
        # Check completeness
        for col in text_columns:
            if col in df.columns:
                completeness = 1 - (df[col].isnull().sum() / len(df))
                score *= completeness
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def validate_embeddings(self, embeddings: Dict[str, np.ndarray]) -> ValidationResult:
        """Validate embeddings quality"""
        errors = []
        warnings = []
        suggestions = []
        
        if not embeddings:
            errors.append("No embeddings provided")
            return ValidationResult(False, errors, warnings, suggestions, 0.0)
        
        # Check embedding dimensions
        dimensions = set()
        for key, embedding in embeddings.items():
            if not isinstance(embedding, np.ndarray):
                errors.append(f"Embedding for '{key}' is not a numpy array")
                continue
            
            dimensions.add(embedding.shape[0])
            
            # Check for NaN or infinite values
            if np.isnan(embedding).any():
                errors.append(f"Embedding for '{key}' contains NaN values")
            if np.isinf(embedding).any():
                errors.append(f"Embedding for '{key}' contains infinite values")
        
        # Check dimension consistency
        if len(dimensions) > 1:
            errors.append(f"Inconsistent embedding dimensions: {dimensions}")
        
        # Check embedding quality
        if embeddings:
            sample_embedding = next(iter(embeddings.values()))
            if np.allclose(sample_embedding, 0):
                warnings.append("Embeddings appear to be all zeros")
        
        # Quality suggestions
        if len(embeddings) < 5:
            suggestions.append("Very few embeddings - consider more data")
        
        quality_score = 1.0 - (len(errors) * 0.4) - (len(warnings) * 0.2)
        quality_score = max(0.0, min(1.0, quality_score))
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, suggestions, quality_score)
    
    def validate_graph_metrics(self, graph_metrics: Dict[str, Any]) -> ValidationResult:
        """Validate graph construction results"""
        errors = []
        warnings = []
        suggestions = []
        
        required_metrics = ['num_nodes', 'num_edges']
        for metric in required_metrics:
            if metric not in graph_metrics:
                errors.append(f"Missing graph metric: {metric}")
        
        if 'num_nodes' in graph_metrics and 'num_edges' in graph_metrics:
            nodes = graph_metrics['num_nodes']
            edges = graph_metrics['num_edges']
            
            if nodes == 0:
                errors.append("Graph has no nodes")
            elif nodes < 3:
                warnings.append("Graph has very few nodes")
            
            if edges == 0:
                warnings.append("Graph has no edges - check similarity threshold")
            
            # Check connectivity
            if edges > 0 and nodes > 0:
                edge_density = edges / (nodes * (nodes - 1) / 2)
                if edge_density < 0.01:
                    suggestions.append("Graph is very sparse - consider lowering similarity threshold")
                elif edge_density > 0.5:
                    suggestions.append("Graph is very dense - consider raising similarity threshold")
        
        quality_score = 1.0 - (len(errors) * 0.5) - (len(warnings) * 0.2)
        quality_score = max(0.0, min(1.0, quality_score))
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, suggestions, quality_score)

# Global validator instance
data_validator = DataValidator()