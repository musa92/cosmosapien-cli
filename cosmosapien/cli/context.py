from ..core.config import ConfigManager
from ..core.model_library import ModelLibrary
from ..core.router import Router

config_manager = ConfigManager()
model_library = ModelLibrary(config_manager)
router = Router(config_manager) 