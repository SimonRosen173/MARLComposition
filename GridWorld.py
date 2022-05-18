import sys
from typing import Tuple, Optional, Union, List, Dict

try:
    from Visualisation import vis
except ImportError as e:
    print(f"vis cannot be imported: {e}")



import gym
import numpy as np
import itertools

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
                 terminal_states: Union[List[int], List[Tuple[int, int]]],
                 joint_start_state: Optional[Union[List[Tuple[int, int]], List[int]]] = None,
                 grid: Optional[Union[np.ndarray, str]] = None,

                 step_reward: float = -0.02,
                 wait_reward: float = - 0.01, wait_at_goal_reward: float = -0.001,
                 collide_reward: float = -1,  # max_steps_reached_reward: float = -2,
                 goal_reward: float = 2, terminal_reward: float = -1,
                 rmin: float = -2,

                 # max_steps: int = 1000,
                 is_flatten_states: float = True,
                 random_starts: float = False,
                 is_warning: bool = True,
                 render_mode: str = "ascii",
                 render_options: Dict = {"tick_time": 0.01}
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

        # self._max_steps = max_steps
        self._curr_step = 0
        self._is_episode_complete = False

        # --------
        #  CHECKS
        # Probs move to a separate method
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

            if type(terminal_states[0]) == int:  # Test
                joint_goals = [np.unravel_index(state, grid.shape) for state in terminal_states]

            if type(joint_goals[0][0]) == int:
                joint_goals = [[np.unravel_index(goal, grid.shape) for goal in joint_goal]
                               for joint_goal in joint_goals]

            print(f"joint_start_state={joint_start_state}")
            print(f"joint_goals={joint_goals}")

        self._joint_goals: List[List[Tuple[int, int]]] = joint_goals
        self._terminal_states: List[Tuple[int, int]] = terminal_states

        if self._random_starts:
            joint_start_state = self._rand_valid_start()

        self._joint_start_state: List[Tuple[int, int]] = joint_start_state

        if is_warning:
            if step_reward > 0:
                print("WARNING: step_reward is positive")
            if wait_reward > 0:
                print("WARNING: wait_reward is positive")
            if collide_reward > 0:
                print("WARNING: collide_reward is positive")
            if terminal_reward > 0:
                print("WARNING: terminal_reward is positive")
            if goal_reward < 0:
                print("WARNING: goal_reward is negative")

        self._step_reward = step_reward
        self._wait_reward = wait_reward
        self._wait_at_goal_reward = wait_at_goal_reward
        self._goal_reward = goal_reward
        self._collide_reward = collide_reward
        self._terminal_reward = terminal_reward
        # self._max_steps_reached_reward = max_steps_reached_reward

        self._joint_state = self._joint_start_state

        # Public variables
        self.rmax = self._goal_reward
        self.rmin = rmin  # min(self._collide_reward, self._step_reward) * 100
        self.diameter = grid.shape[0] + grid.shape[1] - 4

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.grid_shape = grid.shape

        self.trajectory: Optional[List[List[Tuple[int, int]]]] = None

        self._render_mode = render_mode
        self._vis_grid = None

    def _in_bounds(self, coord: Tuple[int, int]):
        x, y = coord
        grid = self._grid
        return (0 <= x < grid.shape[1]) and (0 <= y < grid.shape[0])

    # Check if coord is valid according to grid. I.e., valid if not on obstacle or out of bounds
    def _is_coord_valid(self, coord):
        y, x = coord
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
        assert self._n_agents == 2

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

            # Collision is allowed if at terminal state
            # Only works for 2 agents
            if self._is_collision(joint_state) and joint_state[0] not in self._terminal_states:
                is_valid = False

        return joint_state

    # def _joint_state_intersects_joint_goal(self, joint_state):
    #     for state in joint_state:
    #         if self._is_state_at_goal(state):
    #             return True
    #
    #     return False

    # Check if coord is present in joint goals
    def _is_agent_at_goal(self, state: Tuple[int, int], agent_ind: int):
        assert type(state) == tuple and len(state) == 2

        for joint_goal in self._joint_goals:
            if state == joint_goal[agent_ind]:
                return True

        return False

    # Note: only implemented for 2 agents currently
    def _check_collisions(self, curr_joint_state, next_joint_state):
        if self._n_agents != 2:
            raise NotImplementedError()

        # if one of the agents is at a terminal state there cannot be a collision
        if not (next_joint_state[0] in self._terminal_states or next_joint_state[1] in self._terminal_states):
            # Collision in next_joint_state and not a terminal state
            if next_joint_state[0] == next_joint_state[1]:
                return True, "default"

            # Pass through collision. I.e. agents must pass through each other to reach next state
            if next_joint_state[0] == curr_joint_state[1] and next_joint_state[1] == curr_joint_state[0]:
                return True, "passthrough"

        return False, ""

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
            f"Type mismatch in self._joint_state={self._joint_state}. Must be of form list[tuple[int, int]]."

        n_agents = self._n_agents
        is_done = False
        info = ""

        # reward = self._step_reward
        reward = 0

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

            if action == WAIT:
                if self._is_agent_at_goal(next_state, i):
                    reward += self._wait_at_goal_reward
                else:
                    reward += self._wait_reward
            else:
                reward += self._step_reward

            # If next state is not valid then agent does not move
            if not self._is_coord_valid(next_state):
                next_state = curr_state
                info += f"wall/obstacle collision for agent {i}, "
                # state_invalid = True

            # if curr_state in self._terminal_states:
            #     next_state = curr_state
            #     info += f"agent {i} is in terminal state and cannot move, "

            next_joint_state.append(next_state)

        # agent_collision = False
        # # Check for agent collisions. If there is a collision between agents, and they are not at a terminal state,
        # # then they cannot move and must go to original state.
        # for i in range(0, n_agents):
        #     for j in range(i+1, n_agents):
        #         # Collisions are allowed at terminal states
        #         if next_joint_state[i] == next_joint_state[j] and next_joint_state[i] not in self._terminal_states:
        #             next_joint_state[i] = self._joint_state[i]
        #             next_joint_state[j] = self._joint_state[j]
        #             agent_collision = True
        #
        #             reward += self._collide_reward

        agent_collision, collision_type = self._check_collisions(self._joint_state, next_joint_state)

        if agent_collision:
            # This is only valid for 2 agents
            reward += self._collide_reward
            next_joint_state = self._joint_state
            info += f"agent collision ({collision_type}) outside terminal state, "

        # NOTE: Episode is only completed once joint wait action taken
        if joint_action == [WAIT for _ in range(n_agents)]:
            if next_joint_state in self._joint_goals:
                is_done = True
                reward = self._goal_reward
                info += f"[EPISODE COMPLETE] joint goal reached and joint 'wait' action taken"
            else:
                is_terminal = True
                # joint_state is terminal if all states are a terminal state
                for state in next_joint_state:
                    if state not in self._terminal_states:
                        is_terminal = False

                if is_terminal:
                    is_done = True
                    reward = self._terminal_reward
                    info += f"[EPISODE COMPLETE] joint 'wait' action taken at non-goal terminal joint-state"

        self._joint_state = next_joint_state

        return next_joint_state, reward, is_done, info

    def _flatten_joint_state(self, joint_state: List[Tuple[int, int]]) -> List[int]:
        flat_joint_state: List[int] = [int(np.ravel_multi_index((y, x), self._grid.shape)) for (x, y)
                                       in joint_state]
        return flat_joint_state

    def reset(self, joint_start_state=None, **kwargs) -> Union[List[Tuple[int, int]], List[int]]:
        self._curr_step = 0

        if self._random_starts:
            self._joint_start_state = self._rand_valid_start()

        if "joint_start_state" in kwargs.keys():
            joint_start_state = kwargs["joint_start_state"]

        if joint_start_state is not None:
            assert type(joint_start_state) == list and type(joint_start_state[0]) == tuple, \
                f"Type mismatch in joint_start_start={joint_start_state}. Must be of form list[tuple[int, int]]."
            self._joint_start_state = joint_start_state

        self.trajectory = [self._joint_start_state]

        self._joint_state = self._joint_start_state
        if self._is_flatten_states:
            return self._flatten_joint_state(self._joint_state)
        else:
            return self._joint_state

    def step(self, joint_action: Union[List[int], int], is_action_flattened: bool = False) \
            -> Tuple[list[int], float, bool, str]:
        #
        self._curr_step += 1

        if is_action_flattened:
            assert self._n_agents == 2
            joint_action = list(np.unravel_index(joint_action, (self._n_actions, self._n_actions)))

        next_state, reward, is_done, info = self._take_joint_action(joint_action)

        # if self._curr_step > self._max_steps:
        #     is_done = True
        #     reward = self._max_steps_reached_reward
        #     info += f"[EPISODE COMPLETE] max steps reached"

        self.trajectory.append(next_state)

        if self._is_flatten_states:
            next_state = self._flatten_joint_state(next_state)

        return next_state, reward, is_done, info

    def render(self, mode="ascii", render_options={"tick_time":0.1}):
        grid = self._grid
        joint_goals = self._joint_goals
        joint_state = self._joint_state

        if mode == "canvas":
            if self._vis_grid is None:
                self._vis_grid = vis.VisGrid(grid, (400, 400), 25,
                                             joint_goals=joint_goals, tick_time=render_options["tick_time"])

                self._vis_grid.update_2_agents_pos(joint_state)
                # # TEMP
                self._vis_grid.window.getMouse()
            else:
                self._vis_grid.update_2_agents_pos(joint_state)

            # self._vis_grid.window.getMouse()

        elif mode == "ascii":
            obstacle_char = "#"
            goal_char = "G"
            terminal_char = "T"
            grid_str_arr: List[List[str]] = [[" " for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

            # Obstacles
            for y in range(grid.shape[1]):
                for x in range(grid.shape[0]):
                    if grid[y][x] == 1:
                        grid_str_arr[y][x] = obstacle_char

            for terminal_state in self._terminal_states:
                y, x = terminal_state
                grid_str_arr[y][x] = terminal_char

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

    def get_grid(self):
        return self._grid
    # def set_start_state(self, start_state, is_flat):
    #


class MA4Rooms(MAGridWorld):
    TLC = (3, 3)
    TRC = (3, 9)
    BLC = (9, 3)
    BRC = (9, 9)

    CORRIDOR_TLC = (2, 3)
    CORRIDOR_TRC = (2, 9)
    CORRIDOR_BLC = (10, 3)
    CORRIDOR_BRC = (10, 9)

    TL_CNR = (1, 1)
    TR_CNR = (1, 11)
    BL_CNR = (11, 1)
    BR_CNR = (11, 11)

    def __init__(self, n_agents: int, n_actions: int,
                 rooms_type: str = "default",
                 **kwargs
                 # States can be represented in flattened form (int index) or non-flattened (y, x) form
                 # joint_goals: Union[List[List[int]], List[List[Tuple[int, int]]]],  # flattened or not flattened
                 # joint_start_state: Optional[Union[List[Tuple[int, int]], List[int]]] = None,
                 # step_reward: float = -0.01,
                 # collide_reward: float = -1,
                 # goal_reward: float = 2, terminal_reward: float = -1,
                 # is_flatten_states: bool = True,
                 # random_starts: bool = False,
                 ):

        grid_str = ""
        if rooms_type == "default":
            grid_str = \
                "1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
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
        elif rooms_type == "corridors":
            grid_str = \
                "1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
                "1 0 0 0 0 1 1 1 0 0 0 0 1\n" \
                "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
                "1 0 0 0 0 1 1 1 0 0 0 0 1\n" \
                "1 1 0 1 1 1 1 1 0 0 0 0 1\n" \
                "1 1 0 1 1 1 1 1 1 1 0 1 1\n" \
                "1 1 0 1 1 1 1 1 1 1 0 1 1\n" \
                "1 1 0 1 1 1 1 1 1 1 0 1 1\n" \
                "1 1 0 1 1 1 1 1 1 1 0 1 1\n" \
                "1 0 0 0 0 1 1 1 0 0 0 0 1\n" \
                "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
                "1 0 0 0 0 1 1 1 0 0 0 0 1\n" \
                "1 1 1 1 1 1 1 1 1 1 1 1 1"
            # TLC - (2, 3)
            # TRC - (2, 9)
            # BLC - (10, 3)
            # BRC - (10, 9)

        else:
            raise NotImplementedError()

        grid = [[int(el) for el in row.split(" ")] for row in grid_str.split("\n")]
        grid = np.asarray(grid)

        if rooms_type == "corridors":
            terminal_states = [MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_TRC,
                               MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC]
        else:
            terminal_states = [MA4Rooms.TLC, MA4Rooms.TRC, MA4Rooms.BLC, MA4Rooms.BRC]

        super().__init__(n_agents, n_actions, terminal_states=terminal_states, grid=grid, **kwargs)
        # joint_goals=joint_goals, joint_start_state=joint_start_state,
        # ,
        # step_reward=step_reward, collide_reward=collide_reward,
        # goal_reward=goal_reward, terminal_reward=terminal_reward,
        # is_flatten_states=is_flatten_states, random_starts=random_starts)


# Hacky class to make it look single agent
class MA4RoomsWrapper:
    def __init__(self, ma_4rooms_env: MA4Rooms):
        env = ma_4rooms_env
        self._env = env

        self._n_agents = env.n_agents
        self._n_actions = env.n_actions
        self._grid_shape = env.grid_shape

        self._n_states = env.grid_shape[0] * env.grid_shape[1]

        self._joint_action_shape = tuple([self._n_actions for _ in range(self._n_agents)])
        self._joint_state_shape = tuple([self._n_states for _ in range(self._n_agents)])

        assert len(self._grid_shape) == 2  # Dunno how it wouldn't be tbh

        self.observation_space = gym.spaces.Discrete((self._grid_shape[0]*self._grid_shape[1])**self._n_agents)
        self.action_space = gym.spaces.Discrete(self._n_actions**self._n_agents)

        self.rmin = env.rmin
        self.rmax = env.rmax
        self.diameter = env.diameter

    def step(self, action: int) -> Tuple[int, float, bool, str]:

        joint_action = list(np.unravel_index(action, self._joint_action_shape))
        next_joint_state, reward, is_done, info = self._env.step(joint_action)

        next_state = int(np.ravel_multi_index(next_joint_state, self._joint_state_shape))
        return next_state, reward, is_done, info

    def reset(self, **kwargs) -> int:
        joint_state = self._env.reset(**kwargs)
        state = int(np.ravel_multi_index(joint_state, self._joint_state_shape))
        return state

    def render(self, mode="ascii"):
        self._env.render(mode=mode)

    # Should just do a call to associated method in GridWorld env
    def get_trajectory(self, is_delimited=False):
        if is_delimited:
            return self._delimitate_trajectory()
        else:
            return self._env.trajectory

    def _delimitate_trajectory(self):
        del_str = "|".join([";".join([",".join([str(el) for el in state]) for state in joint_state])
                            for joint_state in self._env.trajectory])
        return del_str

    def get_grid(self):
        return self._env.get_grid()


def main():
    # joint_start_state = [(1, 1), (1, 4)]
    # joint_goals = [[(1, 1), (1, 4)]]

    grid = np.asarray([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])

    joint_start_state = [(0, 2), (2, 0)]
    # noinspection PyListCreation
    joint_goals = [[(0, 0), (2, 2)], [(2, 2), (0, 0)]]  # , [(0, 0), (0, 0)]
    # joint_goals.append([(0, 0), (0, 0)])

    terminal_states = [(0, 0), (0, 2), (2, 0), (2, 2)]

    grid_world = MAGridWorld(n_agents=2, n_actions=N_ACTIONS,
                             grid=grid,
                             joint_start_state=joint_start_state, joint_goals=joint_goals,
                             terminal_states=terminal_states,
                             is_flatten_states=True)

    curr_state = grid_world.reset()
    print(curr_state)
    grid_world.render()

    next_state, reward, is_done, info = grid_world.step([LEFT, UP])
    print(f"{next_state, reward, is_done, info}")
    grid_world.render()

    next_state, reward, is_done, info = grid_world.step([LEFT, UP])
    print(f"{next_state, reward, is_done, info}")

    # next_state, reward, is_done, info = grid_world.step([WAIT, WAIT])
    next_state, reward, is_done, info = grid_world.step(24, is_action_flattened=True)
    print(f"{next_state, reward, is_done, info}")
    grid_world.render()


def test_ma4rooms():
    joint_goals = [[MA4Rooms.TLC, MA4Rooms.BRC]]
    joint_start_state = [(1, 1), (11, 11)]
    env = MA4Rooms(n_agents=2, n_actions=5,
                   joint_goals=joint_goals, joint_start_state=joint_start_state, random_starts=True)
    start_state = env.reset()

    print(start_state)
    env.render()


def test_wrapper():
    joint_goals = [[MA4Rooms.TLC, MA4Rooms.BRC]]
    joint_start_state = [(1, 1), (11, 11)]
    env = MA4Rooms(n_agents=2, n_actions=5,
                   joint_goals=joint_goals, joint_start_state=joint_start_state, random_starts=True)
    env = MA4RoomsWrapper(env)

    start_state = env.reset()

    print(start_state)
    env.render()


def test_traj():
    # 6,9;9,7|6,10;9,7|7,10;9,8|8,10;9,7|8,9;9,8|8,9;9,8|8,9;9,8|9,9;9,9|9,9;9,9|9,9;9,9|9,9;9,9|9,9;9,9|9,9;9,9|9,9;9,9|9,9;9,9|9,9;9,9|9,9;9,9|9,9;9,9
    # R  ;W  |D   ;R  |D   ;L  |L   ;R  |W  ;W  |W  ;W  |
    joint_goals = [[MA4Rooms.TLC, MA4Rooms.BRC]]
    joint_start_state = [(6, 9), (9, 7)]
    env = MA4Rooms(n_agents=2, n_actions=5,
                   joint_goals=joint_goals, joint_start_state=joint_start_state, is_flatten_states=False)
    print(env.reset())
    env.render()

    print("###############")
    next_state, reward, is_done, info = env.step([RIGHT, WAIT])
    print(f"{next_state}, {reward}, {is_done}, {info}")
    env.render()

    print("###############")
    next_state, reward, is_done, info = env.step([DOWN, RIGHT])
    print(f"{next_state}, {reward}, {is_done}, {info}")
    env.render()

    print("###############")
    next_state, reward, is_done, info = env.step([DOWN, LEFT])
    print(f"{next_state}, {reward}, {is_done}, {info}")
    env.render()

    print("###############")
    next_state, reward, is_done, info = env.step([LEFT, RIGHT])
    print(f"{next_state}, {reward}, {is_done}, {info}")
    env.render()

    print("###############")
    next_state, reward, is_done, info = env.step([DOWN, RIGHT])
    print(f"{next_state}, {reward}, {is_done}, {info}")
    env.render()

    print("###############")
    next_state, reward, is_done, info = env.step([WAIT, WAIT])
    print(f"{next_state}, {reward}, {is_done}, {info}")
    env.render()


def test_vis():
    joint_goals = [[MA4Rooms.TLC, MA4Rooms.BRC]]
    joint_start_state = [(1, 1), (11, 11)]
    env = MA4Rooms(n_agents=2, n_actions=5,
                   joint_goals=joint_goals, joint_start_state=joint_start_state, random_starts=True)

    new_vis = vis.VisGrid(env._grid, (400, 400), 25, tick_time=0.5)
    new_vis.window.getMouse()
    new_vis.window.close()
    pass

def test_1():
    env_kwargs = {
        "n_agents": 2,
        "n_actions": 5,
        "goal_reward": 2,
        "collide_reward": -0.02,
        "joint_start_state": [(1, 1), (11, 11)],  # It currently doesn't work if this isn't specified
        "random_starts": True
    }
    # maxiters = 200000(
    maxiters = 100000
    EPS = 0.25

    # -------- #
    #  Task A  #
    # -------- #
    # Agent 1 must go to a bottom goal (i.e., BLC or BRC)
    # Agent 2 can go to any goal
    g_bottom = [MA4Rooms.BLC, MA4Rooms.BRC]
    g_all = [MA4Rooms.BLC, MA4Rooms.BRC, MA4Rooms.TLC, MA4Rooms.TRC]
    joint_goals = list(itertools.product(g_bottom, g_all))
    joint_goals = [list(el) for el in joint_goals]

    env_kwargs["joint_goals"] = joint_goals
    env_A = MA4Rooms(**env_kwargs)

    env_A = MA4RoomsWrapper(env_A)


def interactive_env():
    ENV_KWARGS = {
        "n_agents": 2,
        "n_actions": 5,
        "goal_reward": 2,
        "collide_reward": -0.1,
        # "joint_start_state": [MA4Rooms.BR_CNR, MA4Rooms.BL_CNR],
        "joint_start_state": [(10, 4), (10, 8)],
        "random_starts": False,
        "max_steps": 5,
        "rooms_type": "corridors"
    }

    env_kwargs = ENV_KWARGS

    g_all = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC,
             MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_TRC]

    # ENV A
    g_bottom = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC]

    joint_goals = list(itertools.product(g_bottom, g_all))
    joint_goals = [list(el) for el in joint_goals]

    env_kwargs["joint_goals"] = joint_goals

    env = MA4Rooms(**env_kwargs)
    curr_state = env.reset()
    # print(env.reset())
    env.render(mode="canvas")

    str_int_map = {
        "U": UP,
        "D": DOWN,
        "L": LEFT,
        "R": RIGHT,
        "W": WAIT
    }
    input_str = ""
    while True:
        input_str = input("Input joint-action:")

        if input_str.lower() == "exit":
            break
        joint_action = input_str.upper().split(",")
        joint_action = [str_int_map[action] for action in joint_action]

        next_state, reward, is_done, info = env.step(joint_action)
        env.render(mode="canvas")
        print(f"next_state={next_state}, reward={reward}, is_done={is_done}, info={info}")
        print("##############")


if __name__ == "__main__":
    # main()
    # test_ma4rooms()
    interactive_env()
    # test_wrapper()
    # test_traj()
    # test_1()
