from ultralytics import YOLO
import torch
import torch.nn as nn
from cbam import CBAM
import os
import copy

class YOLOv8WithCBAM:
    def __init__(self, model_path, cbam_locations=None):
        """
        Initialize YOLOv8 with CBAM attention modules
        
        Args:
            model_path: path to YOLOv8 model weights
            cbam_locations: list of tuples (module_name, reduction_ratio) where to insert CBAM
                           if None, it will be added after key backbone C2f blocks
        """
        # Load the base model
        self.model = YOLO(model_path)
        self.nn_model = self.model.model
        
        # Print original model info
        print("\nOriginal YOLOv8 model loaded")
        print(f"Parameters: {sum(p.numel() for p in self.nn_model.parameters())}")
        
        # Default CBAM locations at strategic points in the model
        if cbam_locations is None:
            cbam_locations = []
            
            # Dictionary to store module names and their parameters for CBAM insertion
            cbam_modules_dict = {}
            
            # First pass: gather information about all modules
            for name, module in self.nn_model.named_modules():
                # Target specific blocks where attention would be most effective
                if isinstance(module, nn.Conv2d) and module.out_channels >= 64:
                    parent_name = '.'.join(name.split('.')[:-1])
                    if parent_name and parent_name not in cbam_modules_dict:
                        cbam_modules_dict[parent_name] = module.out_channels
            
            # Select strategic locations only - focus on backbone and head
            for name, channels in cbam_modules_dict.items():
                if any(key in name for key in ['backbone.C2f', 'head']):
                    cbam_locations.append((name, channels))
                    print(f"Adding CBAM location: {name} with {channels} channels")
        
        # Insert CBAM modules
        self.insert_cbam_modules(cbam_locations)
        
    def insert_cbam_modules(self, locations):
        """
        Insert CBAM modules at specified locations
        
        Args:
            locations: list of tuples (module_name, channels)
        """
        print("\nINSERTING CBAM MODULES")
        print("-" * 50)
        
        total_cbam_modules = 0
        for module_name, channels in locations:
            # Find the target module
            parts = module_name.split('.')
            current = self.nn_model
            
            # Navigate to the parent of the target module
            for i, part in enumerate(parts):
                try:
                    # If part is an integer, access as list index
                    idx = int(part)
                    current = current[idx]
                except ValueError:
                    # Otherwise access as attribute
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        print(f"Module path not found: {module_name} - cannot find part '{part}'")
                        break
            else:  # This else belongs to the for loop - executes if no break
                # Module found, now wrap it with CBAM
                try:
                    # Print module info before modification
                    print(f"Original module at {module_name}:")
                    print(f"  Type: {type(current).__name__}")
                    print(f"  Parameters: {sum(p.numel() for p in current.parameters())}")
                    
                    # Store original module
                    original_module = current
                    
                    # Find the parent module to replace the current module
                    parent_path = '.'.join(parts[:-1])
                    last_part = parts[-1]
                    parent = self.nn_model
                    
                    if parent_path:
                        for part in parent_path.split('.'):
                            try:
                                idx = int(part)
                                parent = parent[idx]
                            except ValueError:
                                parent = getattr(parent, part)
                    
                    # Create a new sequential module with the original module and CBAM
                    new_module = nn.Sequential(
                        copy.deepcopy(original_module),
                        CBAM(channels)
                    )
                    
                    # Replace the original module with the new one
                    if last_part.isdigit():
                        parent[int(last_part)] = new_module
                    else:
                        setattr(parent, last_part, new_module)
                    
                    # Print confirmation and module info after modification
                    print(f"Added CBAM module after {module_name} with {channels} channels")
                    print(f"  New module parameters: {sum(p.numel() for p in new_module.parameters())}")
                    print(f"  CBAM parameters: {sum(p.numel() for p in new_module[1].parameters())}")
                    print("-" * 30)
                    
                    total_cbam_modules += 1
                except Exception as e:
                    print(f"Failed to add CBAM at {module_name}: {e}")
                    print("-" * 30)
        
        print(f"\nTotal CBAM modules added: {total_cbam_modules}")
        print("-" * 50)
        
        # Verify model structure after modifications
        print("\nVerifying model structure after CBAM integration")
        modified_params = sum(p.numel() for p in self.nn_model.parameters())
        print(f"Total model parameters: {modified_params}")
        print("-" * 50)
    
    def train(self, **kwargs):
        """
        Train the model using the same API as YOLO
        """
        return self.model.train(**kwargs)
    
    def val(self, **kwargs):
        """
        Validate the model using the same API as YOLO
        """
        return self.model.val(**kwargs)
    
    def predict(self, **kwargs):
        """
        Run prediction using the same API as YOLO
        """
        return self.model.predict(**kwargs)
    def export(self, **kwargs):
        """
        Export the model using the same API as YOLO
        """
        return self.model.export(**kwargs)
        
    def info(self, **kwargs):
        """
        Get model info using the same API as YOLO
        """
        return self.model.info(**kwargs)
