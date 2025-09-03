import numpy as np
import pandas as pd
import tqdm
import multiprocessing
from typing import List, Dict, Tuple
import logging
import os
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Define constants and configuration
CONFIG_FILE = 'config.json'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'

# Define logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'experiment_runner.log')),
        logging.StreamHandler()
    ]
)

class ExperimentStatus(Enum):
    PENDING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4

@dataclass
class ExperimentConfig:
    name: str
    parameters: Dict[str, List[float]]
    num_simulations: int
    num_processes: int

class ExperimentResult:
    def __init__(self, config: ExperimentConfig, results: List[float]):
        self.config = config
        self.results = results

class ExperimentRunner:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()
        self.results = []

    def load_config(self) -> List[ExperimentConfig]:
        with open(self.config_file, 'r') as f:
            config_data = json.load(f)
        return [ExperimentConfig(**c) for c in config_data]

    def run_parameter_sweep(self, config: ExperimentConfig) -> ExperimentResult:
        results = []
        for params in self.generate_parameter_combinations(config.parameters):
            result = self.parallel_simulation(config, params)
            results.append(result)
        return ExperimentResult(config, results)

    def generate_parameter_combinations(self, parameters: Dict[str, List[float]]) -> List[Dict[str, float]]:
        import itertools
        param_names = list(parameters.keys())
        param_values = list(parameters.values())
        combinations = itertools.product(*param_values)
        return [{name: value for name, value in zip(param_names, combination)} for combination in combinations]

    def parallel_simulation(self, config: ExperimentConfig, params: Dict[str, float]) -> float:
        def simulate(params: Dict[str, float]) -> float:
            # Simulate the experiment
            # Replace this with your actual simulation code
            import time
            time.sleep(1)
            return np.random.rand()

        with multiprocessing.Pool(processes=config.num_processes) as pool:
            results = []
            for _ in range(config.num_simulations):
                result = pool.apply_async(simulate, (params,))
                results.append(result.get())
            return np.mean(results)

    def collect_results(self) -> List[ExperimentResult]:
        return self.results

    def save_experiment_data(self, results: List[ExperimentResult]) -> None:
        import pickle
        with open(os.path.join(RESULTS_DIR, 'experiment_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

    def run_experiments(self) -> None:
        for config in self.config:
            try:
                result = self.run_parameter_sweep(config)
                self.results.append(result)
            except Exception as e:
                logging.error(f'Error running experiment {config.name}: {str(e)}')
        self.save_experiment_data(self.collect_results())

if __name__ == '__main__':
    experiment_runner = ExperimentRunner()
    experiment_runner.run_experiments()