from typing import Tuple, List
from collections import defaultdict
 
import copy
import random as rd
import numpy as np
import matplotlib.pyplot as plt


class Maze:
    def __init__(self, R: np.ndarray, end: List[Tuple[int, int]], start: Tuple[int, int]):
        """The init function.
        
        args:
            R (np.ndarray): The array with the rewards per state.
            end (List[Tuple[int, int]]): A list with the positions of the end states.
            start (Tuple[int, int]): The start state of the environment.
        """
        self.R = R
        self.end_states = end
        self.start_state = start
        self.states = np.arange(self.R.size).reshape(self.R.shape)
    
    def get_next_action_positions(self, pos: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Get the next positions of the actions based on the current position.
        
        args:
            pos (Tuple[int, int]): The position for which to decide the next action positions.
        
        returns:
            Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]: The positions of the up, right, left and down actions.
        """
        row, col = pos

        up = (row - 1, col) if row - 1 >= 0 else pos
        right = (row, col + 1) if col + 1 < self.R.shape[1] else pos
        left = (row, col -1) if col - 1 >= 0 else pos
        down = (row + 1, col) if row + 1 < self.R.shape[0] else pos
        return up, right, left, down
    
    def get_random_position(self) -> Tuple[int, int]:
        """Returns a random position in the environment.
        
        returns:
            Tuple[int, int]: A random position in the environment.
        """
        return rd.randint(0, self.R.shape[1] - 1), rd.randint(0, self.R.shape[0] - 1)


def value_iteration(env: Maze, p_action: float = 0.7, discount: float = 0.9, threshold: float = 0.0001) -> np.ndarray:
    """A method for performing value iteration.
    
    args:
        env (Maze): The environment which holds the rewards of all the possible states.
        p_action (float, optional): The probability of a succesfull action. Defaults to 0.7.
        discount (float, optional): The discount factor. Defaults to 0.9.
        threshold (float, optional): The threshold for when the iteration is finished. Defaults to 0.0001.
    
    returns:
        np.ndarray: Returns the utility of every state in the environment.
    """
    
    utility = np.zeros(env.R.shape)
    delta = np.inf
    
    # get all positions in the grid
    positions = [(i, j) for i in range(env.R.shape[0]) for j in range(env.R.shape[1])]
        
    while delta > threshold:
        delta = 0
        new_utility = np.zeros(utility.shape)
        for pos in positions:
            # check if we are evaluating an end state
            if pos in env.end_states:
                # current position is an end-state so value is 0
                continue

            # save the current value
            value = utility[pos]
            # get the next positions of all the actions that can be taken on the current positions
            actions = env.get_next_action_positions(pos)
            action_values = []
            for index, action in enumerate(actions):
                noise_actions = list(actions[:index] + actions[index+1:])
                action_values.append(calculate_action_value(env, utility, p_action, discount, action, noise_actions))

            # select the action with the highest utility
            highest_utility = max(action_values)
            new_utility[pos] = highest_utility
            # update the delta
            delta = max(delta, abs(value - highest_utility))

        utility = copy.deepcopy(new_utility)
    
    return utility


def calculate_action_value(env: Maze, values: np.ndarray, p_action: float, discount: float, action: Tuple[int, int], other_actions: List[Tuple[int, int]]) -> float:
    """Calculates the action value of a certain action with the succesfullness probability of the actions.
    
    args:
        env (Maze): The environment which holds the rewards of all the possible states.
        values (np.ndarray): The array with the values of the states.
        p_action (float): The probability of a succesfull action.
        discount (float): The discount factor.
        action (Tuple[int, int]): The main action that will be taken.
        other_actions (List[Tuple[int, int]]): The other actions that can be taken with the noise in play.
    
    returns:
        float: Returns the value of the action.
    """
    total_value = p_action * (env.R[action] + (discount * values[action]))

    dist_chance = (1 - p_action) / len(other_actions)

    for noise_action in other_actions:
        total_value += dist_chance * (env.R[noise_action] + (discount * values[noise_action]))
    return total_value


def show_utility(values: np.ndarray):
    """Prints the utility array to the screen.
    
    args:
        values (np.ndarray): The array with the values of the states.
    """
    row_divider = "-" * ((8 * values.shape[0]) + values.shape[0] + 1)
    for row in range(values.shape[0]):
        print(row_divider)
        out = "| "
        for col in range(values.shape[1]):
            out += str(round(values[(row, col)], 2)).ljust(6) + ' | '
        print(out)
    print(row_divider)


def show_policy(env: Maze, q_table: defaultdict):
        """Prints the q-table in a matplotlib figure.
        
        args:
            env (Maze): The environment which holds the rewards of all the possible states.
            q_table (defaultdict): The Q-table with the policy.
        """
        plt.rc('figure', figsize=(12,6))
        for row in range(env.R.shape[0]):
            for col in range(env.R.shape[1]):
                # set the x and y value for the place of the state in the figure
                x, y = (col, env.R.shape[1] - row)
                if (row, col) in env.end_states:
                    # end state so draw a point
                    plt.scatter(x, y, s=40)
                else:
                    values = list(map(lambda x: round(x, 3), q_table[(row, col)]))
                    max_elem = max(values)
                    for value, direction in zip(values, ["up", "right", "left", "down"]):
                        if value == max_elem:
                            color = "C3"
                        else:
                            color = "C0"

                        if direction == "up":
                            plt.arrow(x, y, dx=0, dy=0.15, width=0.02, color=color)
                            plt.text(x-0.05, y+0.3, str(value))
                        if direction == "right":
                            plt.arrow(x, y, dx=0.15, dy=0, width=0.02, color=color)
                            plt.text(x+0.3, y, str(value))
                        if direction == "left":
                            plt.arrow(x, y, dx=-0.15, dy=0, width=0.02, color=color)
                            plt.text(x-0.4, y, str(value))
                        if direction == "down":
                            plt.arrow(x, y, dx=0, dy=-0.15, width=0.02, color=color)
                            plt.text(x-0.05, y-0.35, str(value))
        plt.axis("off")
        plt.show()
