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
    def __init__(self, n_agents: int, n_actions: int,
                 # States can be represented in flattened form (int index) or non-flattened (y, x) form
                 joint_goals: Union[List[List[int]], List[List[Tuple[int, int]]]],  # flattened or not flattened
                 joint_start_state: Optional[Union[List[Tuple[int, int]], List[int]]] = None,
                 grid: Optional[Union[np.ndarray, str]] = None,
                 step_reward: bool = -0.01, collide_reward: bool = -1, goal_reward: bool = 1,
                 is_flatten_states: bool = True,
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

        self._is_flatten_states = is_flatten_states
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
                      # Note: PyCharm type checking is being dumb here
                      (type(joint_goals[0][0]) == tuple and len(joint_goals[0][0]) == 2
                       and type(joint_goals[0][0][0]) == int and type(joint_goals[0][1][0]) == int)):
                raise TypeError(type_err_msg)

        #  END OF CHECKS
        # ---------------

        # Convert flat states to non-flattened for joint_goals and joint_start_state
        # This is required for how operations are applied under the hood
        if self._is_flatten_states:
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

    # TODO: Test
    def _is_collision(self, joint_coords):
        for i in range(0, self._n_agents):
            for j in range(i+1, self._n_agents):
                if joint_coords[i] == joint_coords[j]:
                    return True
        return False

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

    def _joint_state_intersects_joint_goal(self, joint_state):
        for state in joint_state:
            if self._is_state_at_goal(state):
                return True

        return False

    # Check if coord is present in joint goals
    def _is_state_at_goal(self, state):
        assert type(state) == tuple and len(state) == 2

        for joint_goal in self._joint_goals:
            if state in joint_goal:
                return True

        return False

    # def _join_state_in_joint_goal(self, joint_state):
    #     pass

    # Apply dynamics
    # Return [next_joint_state, reward, is_done, info]
    # Need to rework this for more than 2 agents
    def _take_joint_action(self, joint_action) \
            -> Tuple[List[Tuple[int, int]], float, bool, str]:
        # Need to figure out how I will set rewards for collisions for multiple agents,
        # e.g. for 3 agents does 2 collisions mean 'reward = 2 * collision_reward' or 'reward = collision_reward'?
        # This may be an issue wrt to composition stuff, but I need to investigate
        if self._n_agents != 2:
            raise NotImplementedError("This method is currently only implemented for 2 agents")

        next_joint_state: List[Tuple[int, int]] = []
        assert type(self._joint_state) == list and type(self._joint_state[0]) == tuple, \
            "Type mismatch in self._joint_state. Must be of form list[tuple[int, int]]"

        n_agents = self._n_agents
        is_done = False
        info = ""
        reward = self._step_reward

        # state_invalid = False

        for i, (curr_state, action) in enumerate(zip(self._joint_state, joint_action)):
            next_state = None
            if action == UP:
                next_state = curr_state[0] - 1, curr_state[1]
            elif action == DOWN:
                next_state = curr_state[0] + 1, curr_state[1]
            elif action == LEFT:
                next_state = curr_state[0], curr_state[1] - 1
            elif action == RIGHT:
                next_state = curr_state[0], curr_state[1] + 1
            elif action == WAIT:
                next_state = curr_state

            # If next state is not valid then agent does not move
            if not self._is_coord_valid(next_state):
                next_state = curr_state
                info += f"wall/obstacle collision for agent {i}, "
                # state_invalid = True

            next_joint_state.append(next_state)

        agent_collision = False
        # This will only work for 2 agents currently -> Very hacky
        # Check for agent collisions. If there is a collision between two agents
        # then they cannot move and must go to original state.
        for i in range(0, n_agents):
            for j in range(i+1, n_agents):
                if next_joint_state[i] == next_joint_state[j]:
                    # Collision is allowed if agents are at joint goal

                    assert self._n_agents == 2  # This current way is super hacky and only works for 2 agents
                    if not [next_joint_state[i], next_joint_state[i]] in self._joint_goals:
                        next_joint_state[i] = self._joint_state[i]
                        next_joint_state[j] = self._joint_state[j]
                        agent_collision = True

        if agent_collision:
            reward = self._collide_reward
            info += "agent collision outside goal, "

        # NOTE: Episode is only completed once
        # If joint wait action taken
        if joint_action == [WAIT for _ in range(n_agents)]:
            if next_joint_state in self._joint_goals:
                is_done = True
                reward = self._goal_reward
                info += f"[EPISODE COMPLETE] goal reached and joint 'wait' action taken"

        self._joint_state = next_joint_state

        return next_joint_state, reward, is_done, info

    def _flatten_joint_state(self, joint_state: List[Tuple[int, int]]) -> List[int]:
        flat_joint_state: List[int] = [int(np.ravel_multi_index((y, x), self._grid.shape)) for (x, y)
                                       in joint_state]
        return flat_joint_state

    def reset(self, *kwargs) -> Union[List[Tuple[int, int]], List[int]]:
        if self._random_starts:
            raise NotImplementedError()

        self._joint_state = self._joint_start_state
        if self._is_flatten_states:
            return self._flatten_joint_state(self._joint_state)
        else:
            return self._joint_state

    def step(self, joint_action: List[int]) -> Tuple[list[int], float, bool, str]:
        next_state, reward, is_done, info = self._take_joint_action(joint_action)
        if self._is_flatten_states:
            next_state = self._flatten_joint_state(next_state)

        return next_state, reward, is_done, info

    def render(self, mode="ascii"):
        obstacle_char = "#"
        goal_char = "G"

        grid = self._grid
        joint_goals = self._joint_goals
        joint_state = self._joint_state

        if mode == "ascii":
            grid_str_arr: List[List[str]] = [[" " for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

            # Obstacles
            for y in range(grid.shape[1]):
                for x in range(grid.shape[0]):
                    if grid[y][x] == 1:
                        grid_str_arr[y][x] = obstacle_char

            # Joint goals
            for joint_goal in joint_goals:
                for goal in joint_goal:
                    y, x = goal
                    grid_str_arr[y][x] = goal_char

            # Agent positions
            for i, (y, x) in enumerate(joint_state):
                grid_str_arr[y][x] = str(i)

            grid_str = "-"*(grid.shape[1]*2 + 1) + "\n"
            grid_str += "\n".join([("|" + " ".join(row) + "|") for row in grid_str_arr])
            grid_str += "\n" + "-"*(grid.shape[1]*2 + 1)
            print(grid_str + "\n")
        else:
            raise NotImplementedError()


def main():
    # joint_start_state = [(1, 1), (1, 4)]
    # joint_goals = [[(1, 1), (1, 4)]]

    grid = np.asarray([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])

    joint_start_state = [(0, 2), (2, 0)]
    # noinspection PyListCreation
    joint_goals = [[(0, 0), (2, 2)], [(2, 2), (0, 0)]]  # , [(0, 0), (0, 0)]
    joint_goals.append([(0, 0), (0, 0)])

    grid_world = MAGridWorld(n_agents=2, n_actions=N_ACTIONS,
                             grid=grid,
                             joint_start_state=joint_start_state, joint_goals=joint_goals, is_flatten_states=True)

    curr_state = grid_world.reset()
    print(curr_state)
    grid_world.render()

    next_state, reward, is_done, info = grid_world.step([LEFT, UP])
    print(f"{next_state, reward, is_done, info}")
    grid_world.render()

    next_state, reward, is_done, info = grid_world.step([LEFT, UP])
    print(f"{next_state, reward, is_done, info}")
    grid_world.render()

    next_state, reward, is_done, info = grid_world.step([WAIT, WAIT])
    print(f"{next_state, reward, is_done, info}")
    grid_world.render()


if __name__ == "__main__":
    main()
