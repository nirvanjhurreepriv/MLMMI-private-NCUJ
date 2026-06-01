import json
import pickle
from pathlib import Path
from typing import Dict, Any

class ModelRegistry:
    def __init__(self, registry_dir: str = "registry"):
        self.registry_dir = Path(registry_dir)
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, dict] = {}
    
    def load_all(self):
        if not self.registry_dir.exists():
            raise FileNotFoundError(f"Registry directory not found: {self.registry_dir}")
        
        for json_file in self.registry_dir.glob("*.json"):
            with open(json_file, "r") as f:
                entry = json.load(f)
            
            model_id = entry["model_id"]
            model_path = Path(entry["model_path"])
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            self.models[model_id] = model
            self.metadata[model_id] = entry
        
        print(f"Loaded {len(self.models)} models from registry")
    
    def get_model(self, model_id: str):
        if model_id not in self.models:
            raise ValueError(f"Unknown model_id: {model_id}. Available: {list(self.models.keys())}")
        return self.models[model_id]
    
    def get_metadata(self, model_id: str):
        return self.metadata.get(model_id)
    
    def list_models(self):
        return list(self.models.keys())