from typing import Tuple, Optional, Union, List

import gym
import numpy as np
from gym.core import ActType, ObsType

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
WAIT = 4
N_ACTIONS = 5


class MAGridWorld(gym.Env):
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) \
            -> Union[ObsType, tuple[ObsType, dict]]:
        pass

    def __init__(self, n_agents: int, n_actions: int,
                 # States can be represented in flattened form (int index) or non-flattened (y, x) form
                 joint_goals: Union[List[List[int]], List[List[Tuple[int, int]]]],  # flattened or not flattened
                 joint_start_state: Optional[Union[List[Tuple[int, int]], List[int]]] = None,
                 grid: Optional[Union[np.ndarray, str]] = None,
                 step_reward: bool = -0.01, collide_reward: bool = -1, goal_reward: bool = 1,
                 flatten_states: bool = True,
                 random_starts: bool = False,
                 ):
        if grid is None:
            # 4 rooms domain -> Probs move to 4 rooms subclass
            grid = "1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 1 0 1 1 1 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 1 1 1 0 1 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
                   "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
                   "1 1 1 1 1 1 1 1 1 1 1 1 1"

        if type(grid) == str:
            grid = [[int(el) for el in row.split(" ")] for row in grid.split("\n")]
            grid = np.asarray(grid)

        self._grid = grid
        self._n_agents = n_agents
        self._n_actions = n_actions

        self._flatten_states = flatten_states
        self._random_starts = random_starts

        # --------
        #  CHECKS
        # joint_start_state
        assert joint_start_state is not None, "Not implemented yet"

        if not type(joint_start_state) == list:
            raise TypeError("joint_start_state must be a List")

        assert len(joint_start_state) == n_agents, "joint_start_state must be of size n_agents"

        if type(joint_start_state) == list:
            pass

        if not (type(joint_start_state[0]) == int or
                # if type of joint_start_state[0] is tuple[int, int]
                (type(joint_start_state[0]) == tuple and len(joint_start_state[0]) == 2
                 and type(joint_start_state[0][0]) == int and type(joint_start_state[0][1]) == int)):
            raise TypeError("joint_start_state must be of type list[tuple[int, int]] or list[int]")

        # joint_goals
        if type(joint_goals) == list and len(joint_goals) == 0:
            raise ValueError("joint_goals cannot be an empty list")

        type_err_msg = "joint_goals must be of form list[list[int]] or list[list[tuple[int, int]]]"
        if not type(joint_goals) == list:
            raise TypeError(type_err_msg)
        else:
            if len(joint_goals) == 0:
                raise ValueError("joint_goals cannot be an empty list")
            elif not type(joint_goals[0]) == list:
                raise TypeError(type_err_msg)
            elif not len(joint_goals[0]) == n_agents:
                raise ValueError("Each joint_goal in joint_goals must be of size n_agents")
            # Check if joint_goals[0][0] is of type int or type tuple[int, int]
            elif not (type(joint_goals[0][0]) == int or
                      # is of type tuple[int, int]
                      (type(joint_goals[0][0]) == tuple and len(joint_goals[0]) == 2
                       and type(joint_goals[0][0]) == int and type(joint_goals[0][1]) == int)):
                raise TypeError(type_err_msg)

        #  END OF CHECKS
        # ---------------

        # Convert flat states to non-flattened for joint_goals and joint_start_state
        # This is required for how operations are applied under the hood
        if self._flatten_states:
            if type(joint_start_state[0]) == int:
                joint_start_state = [np.unravel_index(start_state, grid.shape) for start_state in joint_start_state]

            if type(joint_goals[0][0]) == int:
                joint_goals = [[np.unravel_index(goal, grid.shape) for goal in joint_goal]
                               for joint_goal in joint_goals]

            print(f"joint_start_state={joint_start_state}")
            print(f"joint_goals={joint_goals}")

        self._joint_goals: List[List[Tuple[int, int]]] = joint_goals
        self._joint_start_state: List[Tuple[int, int]] = joint_start_state

        self._step_reward = step_reward
        self._goal_reward = goal_reward
        self._collide_reward = collide_reward

        self._joint_state = self._joint_start_state

    def _in_bounds(self, coord: Tuple[int, int]):
        x, y = coord
        grid = self._grid
        return (0 <= x < grid.shape[1]) and (0 <= y < grid.shape[0])

    # Check if coord is valid according to grid. I.e., valid if not on obstacle or out of bounds
    def _is_coord_valid(self, coord):
        x, y = coord
        grid = self._grid
        # Is it in bounds of grid
        if (0 <= x < len(grid[1])) and (0 <= y < len(grid)):
            # Is there an obstacle. If state is on obstacle then it cannot be valid
            return grid[y][x] == 0
        else:
            return False

    # TODO: Implement
    def _is_collision(self, joint_coords):
        raise NotImplementedError()

    # TODO: Test
    def _rand_valid_start(self):
        is_valid = False
        y_len, x_len = self._grid.shape
        joint_state = None
        while not is_valid:
            joint_state = [(np.random.randint(0, y_len), np.random.randint(0, x_len)) for _ in range(self._n_agents)]
            is_valid = True

            for state in joint_state:
                if not self._is_coord_valid(state):
                    is_valid = False
                    break

            if self._is_collision(joint_state):
                is_valid = False

        return joint_state

    def _take_joint_action(self, joint_action):
        assert self._n_actions == 5

        next_joint_state = []
        n_agents = self._n_agents
        is_done = False
        info = ""

        state_invalid = False
        for i, (curr_state, action) in enumerate(zip(self._joint_state, joint_action)):
            next_state = None
            if action == UP:
                next_state = curr_state[0], curr_state[1] - 1
            elif action == DOWN:
                next_state = curr_state[0], curr_state[1] + 1
            elif action == LEFT:
                next_state = curr_state[0] - 1, curr_state[1]
            elif action == RIGHT:
                next_state = curr_state[0] + 1, curr_state[1]

            # If next state is not valid then agent does not move
            if not self._is_state_valid(next_state):
                next_state = curr_state
                state_invalid = True

            next_joint_state.append(next_state)

        if state_invalid:
            info += "wall/obstacle collision, "

        agent_collision = False
        # Check for agent collisions. If there is a collision between two agents
        # then they cannot move and must go to original state.
        for i in range(0, n_agents):
            for j in range(i+1, n_agents):
                if next_joint_state[i] == next_joint_state[j]:
                    next_joint_state[i] = self.joint_state[i]
                    next_joint_state[j] = self.joint_state[j]
                    agent_collision = True

        # Assign rewards and check if at goal/terminal
        if agent_collision:
            reward = self.collide_reward
            info += "agent collision, "
        else:
            # check if at goal
            if tuple(next_joint_state) in self.joint_goals:
                reward = self.goal_reward
                is_done = True
                info += "joint goal reached, "
            # Check if in terminal state that is not goal
            elif tuple(next_joint_state) in self.joint_terminal_states:
                reward = self.terminal_reward
                is_done = True
                info += "non-goal terminal joint state reached, "
            else:
                reward = self.step_reward

        assert len(next_joint_state) == self.n_agents
        self.joint_state = next_joint_state

        if self.flatten_state:
            flat_joint_state: List[int] = [int(np.ravel_multi_index((y, x), self.grid.shape)) for (x, y)
                                           in self.joint_state]
            return flat_joint_state, reward, is_done, info
        else:
            return next_joint_state, reward, is_done, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:

        pass

    def render(self, mode="human"):
        pass


def main():
    # joint_start_state = [(1, 1), (1, 4)]
    # joint_goals = [[(1, 1), (1, 4)]]

    joint_start_state = [1, 4]
    joint_goals = [[1, 1], [3, 5]]

    grid_world = MAGridWorld(n_agents=2, n_actions=N_ACTIONS,
                             joint_start_state=joint_start_state, joint_goals=joint_goals, flatten_states=True)
    pass


if __name__ == "__main__":
    main()
