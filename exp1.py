from functools import partial

from GridWorld import *
from library import Goal_Oriented_Q_learning, Q_learning, follow_extended_q_policy, follow_q_policy
# extended_q_dict_to_numpy
from library import OR, AND, NOT
import benchmark

import itertools


def main():
    # env = MA4Rooms(n_agents=2, n_actions=5,
    #                goal_reward=2, collide_reward=-0.02,
    #                joint_goals=joint_goals, joint_start_state=joint_start_state, random_starts=True)
    env_kwargs = {
        "n_agents": 2,
        "n_actions": 5,
        "goal_reward": 2,
        "collide_reward": -0.02,
        "joint_start_state": [(1, 1), (11, 11)],  # It currently doesn't work if this isn't specified
        "random_starts": True
    }
    # maxiters = 200000(
    maxiters = 10000
    hyper_params = {
        "maxiter": maxiters,

        "epsilon": 0.25,  # Will be ignored if is_eps_decay = True

        "is_printing": True,

        "is_eps_decay": True,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_rate": 1.0/maxiters
    }

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

    Q_A, stats_A = Goal_Oriented_Q_learning(env_A, **hyper_params)

    all_test_starts = [MA4Rooms.TL_CNR, MA4Rooms.TR_CNR, MA4Rooms.BL_CNR, MA4Rooms.BR_CNR]
    all_test_joint_starts = list(itertools.product(all_test_starts, all_test_starts))
    all_test_joint_starts = [list(el) for el in all_test_joint_starts]
    # Remove joint_starts with collisions
    all_test_joint_starts = list(filter(lambda x: x[0] != x[1], all_test_joint_starts))

    # n_states = env_A.observation_space.n
    # n_goals = 16
    # n_actions = env_A.action_space.n
    # Q_A_arr = extended_q_dict_to_numpy(Q_A, n_states, n_goals, n_actions)

    benchmark.follow_policies(env_A, Q_A, all_test_joint_starts, "trajs/Q_A_traj.txt")

    # -------- #
    #  Task B  #
    # -------- #
    # Agent 1 can go to any goal
    # Agent 2 must go to a right goal (i.e., TRC or BRC)
    g_right = [MA4Rooms.TRC, MA4Rooms.BRC]
    joint_goals = list(itertools.product(g_all, g_right))
    joint_goals = [list(el) for el in joint_goals]

    env_kwargs["joint_goals"] = joint_goals
    env_B = MA4Rooms(**env_kwargs)

    env_B = MA4RoomsWrapper(env_B)

    Q_B, stats_B = Goal_Oriented_Q_learning(env_B, **hyper_params)

    benchmark.follow_policies(env_B, Q_B, all_test_joint_starts, "trajs/Q_B_traj.txt")

    # -------------------------- #
    #  Task (NOT A) AND (NOT B)  #
    # -------------------------- #
    # Agent 1 must go to a top goal (i.e., TLC, TRC)
    # Agent 2 must go to a left goal (i.e., TLC, BLC)

    n_actions = env_B.action_space.n

    AND_PTL = partial(AND, n_actions=n_actions)
    NOT_PTL = partial(NOT, n_actions=n_actions, rmin=env_A.rmin, rmax=env_A.rmax)

    print("# A AND B #")
    Q_comp_and = AND_PTL(Q_A, Q_B)

    # env shouldn't really matter here since dynamics are the same and only the rewards are different
    benchmark.follow_policies(env_A, Q_comp_and, all_test_joint_starts, "trajs/Q_comp_and_traj.txt")

    Q_comp = AND_PTL(NOT_PTL(Q_A), NOT_PTL(Q_B))
    # # env shouldn't really matter here since dynamics are the same and only the rewards are different
    benchmark.follow_policies(env_A, Q_comp, all_test_joint_starts, "trajs/Q_comp_traj.txt")


if __name__ == '__main__':
    main()
