import yaml
import numpy as np
from pathlib import Path
from warnings import warn

class SimulationManager:
    def __init__(self, config_path=None, controller=None, environment=None,
                 reproduction=None, reporters=None, visualizer=None):
        if config_path is None:
            raise ValueError("Configuration path must be provided.")
        if controller is None:
            raise ValueError("Controller must be provided.")
        if environment is None:
            raise ValueError("Environment must be provided.")
        if reproduction is None:
            raise ValueError("Reproduction method must be provided.")
        if reporters is None:
            warn("Reporters are not provided, no data will be saved.", UserWarning)
        if visualizer is None:
            warn("Visualizers are not provided, defaulting to headless mode.", UserWarning)
        self.cfg = yaml.safe_load(Path(config_path).read_text())
        np.random.seed(self.cfg['general']['seed'])