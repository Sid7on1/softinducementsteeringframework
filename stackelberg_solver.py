import numpy as np
from scipy.optimize import linprog
import cvxpy as cp
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StackelbergSolver:
    """
    Computes Stackelberg equilibria for the mediator's optimization problem given game parameters.
    """

    def __init__(self, game_params, mediator_params):
        """
        Initializes the Stackelberg solver with game and mediator parameters.

        Args:
            game_params (dict): Game parameters, including the number of players, actions, and payoffs.
            mediator_params (dict): Mediator parameters, including the objective function and constraints.
        """
        self.game_params = game_params
        self.mediator_params = mediator_params

    def check_case_conditions(self):
        """
        Checks if the game parameters satisfy the case conditions for the Stackelberg equilibrium.

        Returns:
            bool: True if the case conditions are satisfied, False otherwise.
        """
        # Check if the number of players is 2
        if self.game_params['num_players'] != 2:
            logger.error("The number of players must be 2.")
            return False

        # Check if the number of actions is 2
        if self.game_params['num_actions'] != 2:
            logger.error("The number of actions must be 2.")
            return False

        # Check if the payoffs are symmetric
        if not np.allclose(self.game_params['payoffs'], self.game_params['payoffs'].T):
            logger.error("The payoffs must be symmetric.")
            return False

        return True

    def calculate_mediator_utility(self, mediator_action):
        """
        Calculates the mediator's utility given the mediator's action.

        Args:
            mediator_action (float): The mediator's action.

        Returns:
            float: The mediator's utility.
        """
        # Calculate the mediator's utility using the objective function
        mediator_utility = self.mediator_params['objective_function'](mediator_action)
        return mediator_utility

    def solve_lower_level_lp(self, mediator_action):
        """
        Solves the lower-level linear program given the mediator's action.

        Args:
            mediator_action (float): The mediator's action.

        Returns:
            tuple: The solution to the lower-level linear program.
        """
        # Define the lower-level linear program
        c = np.array([-self.game_params['payoffs'][0, 0], -self.game_params['payoffs'][0, 1]])
        A_ub = np.array([[1, 1]])
        b_ub = np.array([1])
        A_eq = np.array([[1, 0], [0, 1]])
        b_eq = np.array([1, 1])

        # Solve the lower-level linear program
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')

        # Return the solution to the lower-level linear program
        return res.x, res.fun

    def compute_stackelberg_equilibrium(self):
        """
        Computes the Stackelberg equilibrium for the mediator's optimization problem.

        Returns:
            tuple: The Stackelberg equilibrium, including the mediator's action and the players' actions.
        """
        # Check if the case conditions are satisfied
        if not self.check_case_conditions():
            logger.error("The case conditions are not satisfied.")
            return None

        # Initialize the mediator's action
        mediator_action = 0.0

        # Initialize the players' actions
        players_actions = np.array([0.0, 0.0])

        # Initialize the mediator's utility
        mediator_utility = 0.0

        # Iterate over the possible mediator actions
        for i in range(100):
            # Calculate the mediator's utility given the current mediator action
            mediator_utility = self.calculate_mediator_utility(mediator_action)

            # Solve the lower-level linear program given the current mediator action
            players_actions, _ = self.solve_lower_level_lp(mediator_action)

            # Update the mediator's action
            mediator_action = (1 - 0.1) * mediator_action + 0.1 * players_actions[0]

        # Return the Stackelberg equilibrium
        return mediator_action, players_actions

def main():
    # Set up the game parameters
    game_params = {
        'num_players': 2,
        'num_actions': 2,
        'payoffs': np.array([[3, 0], [0, 3]])
    }

    # Set up the mediator parameters
    mediator_params = {
        'objective_function': lambda x: -x
    }

    # Create a Stackelberg solver
    solver = StackelbergSolver(game_params, mediator_params)

    # Compute the Stackelberg equilibrium
    equilibrium = solver.compute_stackelberg_equilibrium()

    # Print the Stackelberg equilibrium
    if equilibrium is not None:
        logger.info("Stackelberg Equilibrium: Mediator's Action = %.2f, Players' Actions = (%.2f, %.2f)", *equilibrium)
    else:
        logger.error("Failed to compute the Stackelberg equilibrium.")

if __name__ == "__main__":
    main()