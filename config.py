import numpy as np
import logging
import json
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigException(Exception):
    """Base exception class for configuration-related errors."""
    pass

class InvalidParameterError(ConfigException):
    """Raised when an invalid parameter is encountered."""
    pass

class Config:
    """Central configuration class for game parameters, algorithm settings, and simulation constants."""

    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the configuration object.

        Args:
        - params (Dict[str, Any]): A dictionary of configuration parameters.
        """
        self.params = params if params else {}
        self.default_params = self.load_default_params()

    def load_default_params(self) -> Dict[str, Any]:
        """
        Load default configuration parameters.

        Returns:
        - Dict[str, Any]: A dictionary of default configuration parameters.
        """
        default_params = {
            "game": {
                "num_players": 2,
                "num_actions": 2,
                "payoff_matrix": np.array([[1, 0], [0, 1]])
            },
            "algorithm": {
                "learning_rate": 0.1,
                "exploration_rate": 0.1,
                "num_iterations": 1000
            },
            "simulation": {
                "num_simulations": 100,
                "seed": 42
            }
        }
        return default_params

    def validate_parameters(self) -> None:
        """
        Validate the configuration parameters.

        Raises:
        - InvalidParameterError: If an invalid parameter is encountered.
        """
        required_params = ["game", "algorithm", "simulation"]
        for param in required_params:
            if param not in self.params:
                raise InvalidParameterError(f"Missing required parameter: {param}")

        if "game" in self.params:
            if "num_players" not in self.params["game"] or "num_actions" not in self.params["game"]:
                raise InvalidParameterError("Missing required game parameter: num_players or num_actions")
            if self.params["game"]["num_players"] < 1 or self.params["game"]["num_actions"] < 1:
                raise InvalidParameterError("Invalid game parameter: num_players or num_actions must be positive")

        if "algorithm" in self.params:
            if "learning_rate" not in self.params["algorithm"] or "exploration_rate" not in self.params["algorithm"]:
                raise InvalidParameterError("Missing required algorithm parameter: learning_rate or exploration_rate")
            if self.params["algorithm"]["learning_rate"] < 0 or self.params["algorithm"]["exploration_rate"] < 0:
                raise InvalidParameterError("Invalid algorithm parameter: learning_rate or exploration_rate must be non-negative")

        if "simulation" in self.params:
            if "num_simulations" not in self.params["simulation"] or "seed" not in self.params["simulation"]:
                raise InvalidParameterError("Missing required simulation parameter: num_simulations or seed")
            if self.params["simulation"]["num_simulations"] < 1:
                raise InvalidParameterError("Invalid simulation parameter: num_simulations must be positive")

    def export_config(self) -> Dict[str, Any]:
        """
        Export the configuration parameters.

        Returns:
        - Dict[str, Any]: A dictionary of configuration parameters.
        """
        return self.params

    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration parameters from a file.

        Args:
        - file_path (str): The path to the configuration file.

        Raises:
        - FileNotFoundError: If the file does not exist.
        - json.JSONDecodeError: If the file is not a valid JSON file.
        """
        try:
            with open(file_path, "r") as file:
                self.params = json.load(file)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {file_path} - {e}")
            raise

    def save_to_file(self, file_path: str) -> None:
        """
        Save the configuration parameters to a file.

        Args:
        - file_path (str): The path to the configuration file.
        """
        with open(file_path, "w") as file:
            json.dump(self.params, file, indent=4)

def main():
    # Example usage
    config = Config()
    config.load_from_file("config.json")
    config.validate_parameters()
    print(config.export_config())

if __name__ == "__main__":
    main()