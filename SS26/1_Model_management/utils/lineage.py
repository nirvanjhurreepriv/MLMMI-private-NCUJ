# utils/lineage.py - Model lineage tracking utilities
"""
Simple but effective lineage tracking for model derivation relationships.

Supports tracking:
- Parent-child model relationships
- Types of derivation: hyperparameter tuning, dataset change, algorithm change
- Descriptive notes for reproducibility

(Inspired by ModelDB lineage concepts)
"""

from typing import Optional, Dict, List


def create_lineage_entry(
    parent_model_id: Optional[str] = None,
    relationship_type: str = "derived_from",
    description: str = "",
    dataset_change: Optional[str] = None,
    hyperparameter_changes: Optional[Dict] = None,
    algorithm_change: Optional[Dict] = None,
    additional_metadata: Optional[Dict] = None
) -> Dict:
    """
    Create a lineage entry describing how a model was derived.
    
    Args:
        parent_model_id: ID of the parent/base model (None if baseline)
        relationship_type: Type of relationship (e.g., 'hyperparameter_tuned', 
                          'retrained_on_new_data', 'algorithm_variant')
        description: Human-readable description of the derivation
        dataset_change: Description of dataset version change (if applicable)
        hyperparameter_changes: Dict describing hyperparameter modifications
        algorithm_change: Dict describing algorithm modifications (if applicable)
        additional_metadata: Any other lineage-relevant metadata
    
    Returns:
        Dict: Lineage entry ready to be stored in model registry
    """
    lineage = {
        "parent_model_id": parent_model_id,
        "relationship_type": relationship_type,
        "description": description
    }
    
    # Add optional fields only if provided (keeps JSON clean)
    if dataset_change:
        lineage["dataset_change"] = dataset_change
    if hyperparameter_changes:
        lineage["hyperparameter_changes"] = hyperparameter_changes
    if algorithm_change:
        lineage["algorithm_change"] = algorithm_change
    if additional_metadata:
        lineage.update(additional_metadata)
    
    return lineage


def get_lineage_chain(registry, model_id: str) -> List[Dict]:
    """
    Trace the full lineage chain for a model back to its root.
    
    Args:
        registry: ModelRegistry instance
        model_id: Starting model ID
    
    Returns:
        List[Dict]: List of lineage entries from root to current model
    """
    chain = []
    current_id = model_id
    
    while current_id:
        try:
            _, metadata = registry.get_model(current_id)
            lineage = metadata.get("lineage", {})
            
            if lineage:
                chain.append({
                    "model_id": current_id,
                    "lineage": lineage
                })
                current_id = lineage.get("parent_model_id")
            else:
                # Root model (no parent)
                chain.append({
                    "model_id": current_id,
                    "lineage": {"is_root": True}
                })
                break
        except FileNotFoundError:
            break
    
    # Reverse to get root -> current order
    return list(reversed(chain))