"""Plugin manager for extending Cosmosapien CLI."""

import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..core.models import BaseModel


class PluginManager:
    """Manages plugins for extending Cosmosapien CLI."""
    
    def __init__(self, plugins_path: str = "~/.cosmosapien/plugins"):
        self.plugins_path = Path(plugins_path).expanduser()
        self.plugins_path.mkdir(parents=True, exist_ok=True)
        self.loaded_plugins: Dict[str, Any] = {}
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin by name."""
        try:
            # Try to load from plugins directory first
            plugin_file = self.plugins_path / f"{plugin_name}.py"
            if plugin_file.exists():
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                # Try to load as a regular module
                module = importlib.import_module(plugin_name)
            
            # Check if module has required attributes
            if hasattr(module, 'setup'):
                module.setup()
            
            self.loaded_plugins[plugin_name] = module
            return True
            
        except Exception as e:
            print(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    def load_all_plugins(self) -> List[str]:
        """Load all plugins from the plugins directory."""
        loaded = []
        
        for plugin_file in self.plugins_path.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            
            plugin_name = plugin_file.stem
            if self.load_plugin(plugin_name):
                loaded.append(plugin_name)
        
        return loaded
    
    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """Get a loaded plugin."""
        return self.loaded_plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all loaded plugins."""
        return list(self.loaded_plugins.keys())
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name in self.loaded_plugins:
            del self.loaded_plugins[plugin_name]
            return True
        return False
    
    def register_model(self, provider_name: str, model_class: type) -> None:
        """Register a custom model from a plugin."""
        from ..core.models import model_registry
        model_registry.register(provider_name, model_class)
    
    def get_plugin_commands(self) -> List[Any]:
        """Get all commands from loaded plugins."""
        commands = []
        
        for plugin_name, plugin in self.loaded_plugins.items():
            if hasattr(plugin, 'commands'):
                commands.extend(plugin.commands)
        
        return commands 