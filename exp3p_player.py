import numpy as np
from scipy.special import softmax
import logging
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EXP3PPlayer:
    """
    Implements EXP3.P no-regret learning algorithm for players with signal-specific instances.

    Attributes:
    - num_actions (int): The number of actions available to the player.
    - learning_rate (float): The learning rate for the EXP3.P algorithm.
    - weights (np.ndarray): The weights for each action.
    - losses (np.ndarray): The losses for each action.
    - gamma (float): The exploration rate.
    - eta (float): The learning rate for the weights.
    """

    def __init__(self, num_actions: int, learning_rate: float, gamma: float, eta: float):
        """
        Initializes the EXP3.P player.

        Args:
        - num_actions (int): The number of actions available to the player.
        - learning_rate (float): The learning rate for the EXP3.P algorithm.
        - gamma (float): The exploration rate.
        - eta (float): The learning rate for the weights.
        """
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eta = eta
        self.weights = np.ones(num_actions)
        self.losses = np.zeros(num_actions)

    def select_action(self) -> int:
        """
        Selects an action using the EXP3.P algorithm.

        Returns:
        - int: The selected action.
        """
        # Calculate the probabilities for each action
        probabilities = softmax(self.weights)
        # Select an action based on the probabilities
        action = np.random.choice(self.num_actions, p=probabilities)
        return action

    def update_weights(self, action: int, loss: float):
        """
        Updates the weights for the EXP3.P algorithm.

        Args:
        - action (int): The action taken.
        - loss (float): The loss for the action.
        """
        # Update the losses for each action
        self.losses[action] = loss
        # Update the weights for each action
        self.weights = self.weights * np.exp(-self.eta * self.losses)
        # Normalize the weights
        self.weights = self.weights / np.sum(self.weights)

    def get_regret_bounds(self) -> Tuple[float, float]:
        """
        Calculates the regret bounds for the EXP3.P algorithm.

        Returns:
        - Tuple[float, float]: The regret bounds.
        """
        # Calculate the regret bounds
        regret_bound = self.learning_rate * np.log(self.num_actions) + self.gamma * self.num_actions
        return regret_bound, regret_bound

    def reset(self):
        """
        Resets the EXP3.P player.
        """
        # Reset the weights and losses
        self.weights = np.ones(self.num_actions)
        self.losses = np.zeros(self.num_actions)

class EXP3PPlayerException(Exception):
    """
    Exception class for EXP3.P player.
    """
    pass

class EXP3PPlayerConfig:
    """
    Configuration class for EXP3.P player.
    """
    def __init__(self, num_actions: int, learning_rate: float, gamma: float, eta: float):
        """
        Initializes the EXP3.P player configuration.

        Args:
        - num_actions (int): The number of actions available to the player.
        - learning_rate (float): The learning rate for the EXP3.P algorithm.
        - gamma (float): The exploration rate.
        - eta (float): The learning rate for the weights.
        """
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eta = eta

def validate_config(config: EXP3PPlayerConfig):
    """
    Validates the EXP3.P player configuration.

    Args:
    - config (EXP3PPlayerConfig): The EXP3.P player configuration.

    Raises:
    - EXP3PPlayerException: If the configuration is invalid.
    """
    if config.num_actions <= 0:
        raise EXP3PPlayerException("Number of actions must be greater than 0")
    if config.learning_rate <= 0:
        raise EXP3PPlayerException("Learning rate must be greater than 0")
    if config.gamma < 0 or config.gamma > 1:
        raise EXP3PPlayerException("Exploration rate must be between 0 and 1")
    if config.eta <= 0:
        raise EXP3PPlayerException("Learning rate for weights must be greater than 0")

def main():
    # Create an EXP3.P player configuration
    config = EXP3PPlayerConfig(num_actions=5, learning_rate=0.1, gamma=0.5, eta=0.01)
    # Validate the configuration
    validate_config(config)
    # Create an EXP3.P player
    player = EXP3PPlayer(config.num_actions, config.learning_rate, config.gamma, config.eta)
    # Select an action
    action = player.select_action()
    logger.info(f"Selected action: {action}")
    # Update the weights
    player.update_weights(action, 0.5)
    # Get the regret bounds
    regret_bound, _ = player.get_regret_bounds()
    logger.info(f"Regret bound: {regret_bound}")
    # Reset the player
    player.reset()

if __name__ == "__main__":
    main()