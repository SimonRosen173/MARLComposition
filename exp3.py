from functools import partial

from GridWorld import *
from library import Goal_Oriented_Q_learning, Q_learning, follow_extended_q_policy, follow_q_policy
import library
from library import OR, AND, NOT
import benchmark
import multiprocessing

import itertools


# def train_task(task_id):
#     base_pkl_path = "Q_pkls/corridors/2/"
#
#     env_kwargs = {
#         "n_agents": 2,
#         "n_actions": 5,
#         "goal_reward": 2,
#         "collide_reward": -0.1,
#         "joint_start_state": [(1, 1), (11, 11)],  # It currently doesn't work if this isn't specified
#         "random_starts": True,
#         "rooms_type": "corridors"
#     }
#
#     maxiters = 1000
#     hyper_params = {
#         "maxiter": maxiters,
#
#         "epsilon": 0.25,  # Will be ignored if is_eps_decay = True
#
#         "is_printing": True,
#
#         "is_eps_decay": True,
#         "eps_start": 1.0,
#         "eps_end": 0.01,
#         "eps_decay_rate": 1.0 / maxiters
#     }
#
#     all_test_starts = [MA4Rooms.TL_CNR, MA4Rooms.TR_CNR, MA4Rooms.BL_CNR, MA4Rooms.BR_CNR]
#     all_test_joint_starts = list(itertools.product(all_test_starts, all_test_starts))
#     all_test_joint_starts = [list(el) for el in all_test_joint_starts]
#     # Remove joint_starts with collisions
#     all_test_joint_starts = list(filter(lambda x: x[0] != x[1], all_test_joint_starts))
#
#     g_all = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC,
#              MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_TRC]
#
#     if task_id == "A":
#         # -------- #
#         #  Task A  #
#         # -------- #
#         # Agent 1 must go to a bottom goal (i.e., BLC or BRC)
#         # Agent 2 can go to any goal
#         g_bottom = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC]
#
#         joint_goals = list(itertools.product(g_bottom, g_all))
#         joint_goals = [list(el) for el in joint_goals]
#
#         env_kwargs["joint_goals"] = joint_goals
#         env_A = MA4Rooms(**env_kwargs)
#
#         env_A = MA4RoomsWrapper(env_A)
#
#         Q_A, stats_A = Goal_Oriented_Q_learning(env_A, print_prefix=task_id, **hyper_params)
#
#         library.save_extended_Q(Q_A, base_pkl_path + "Q_A.pkl")
#
#         benchmark.follow_policies(env_A, Q_A, all_test_joint_starts, "trajs/Q_A_traj.txt")
#
#         return Q_A, stats_A
#     else:
#         # -------- #
#         #  Task B  #
#         # -------- #
#         # Agent 1 can go to any goal
#         # Agent 2 must go to a right goal (i.e., TRC or BRC)
#         g_right = [MA4Rooms.CORRIDOR_TRC, MA4Rooms.CORRIDOR_BRC]
#         joint_goals = list(itertools.product(g_all, g_right))
#         joint_goals = [list(el) for el in joint_goals]
#
#         env_kwargs["joint_goals"] = joint_goals
#         env_B = MA4Rooms(**env_kwargs)
#
#         env_B = MA4RoomsWrapper(env_B)
#
#         Q_B, stats_B = Goal_Oriented_Q_learning(env_B, print_prefix=task_id, **hyper_params)
#         library.save_extended_Q(Q_B, base_pkl_path + "Q_B.pkl")
#
#         benchmark.follow_policies(env_B, Q_B, all_test_joint_starts, "trajs/Q_B_traj.txt")
#
#         return Q_B, stats_B


def main():
    base_pkl_path = "Q_pkls/corridors/4/"

    env_kwargs = {
        "n_agents": 2,
        "n_actions": 5,
        "goal_reward": 2,
        "collide_reward": -0.1,
        "joint_start_state": [(1, 1), (11, 11)],  # It currently doesn't work if this isn't specified
        "random_starts": True,
        "rooms_type": "corridors"
    }

    maxiters = 50000
    hyper_params = {
        "maxiter": maxiters,

        "epsilon": 0.25,  # Will be ignored if is_eps_decay = True

        "is_printing": True,

        "is_eps_decay": True,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_rate": 1.0 / maxiters
    }

    all_test_starts = [MA4Rooms.TL_CNR, MA4Rooms.TR_CNR, MA4Rooms.BL_CNR, MA4Rooms.BR_CNR]
    all_test_joint_starts = list(itertools.product(all_test_starts, all_test_starts))
    all_test_joint_starts = [list(el) for el in all_test_joint_starts]
    # Remove joint_starts with collisions
    all_test_joint_starts = list(filter(lambda x: x[0] != x[1], all_test_joint_starts))

    g_all = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC,
             MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_TRC]

    # -------- #
    #  Task A  #
    # -------- #
    # Agent 1 must go to a bottom goal (i.e., BLC or BRC)
    # Agent 2 can go to any goal
    g_bottom = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC]

    joint_goals = list(itertools.product(g_bottom, g_all))
    joint_goals = [list(el) for el in joint_goals]

    env_kwargs["joint_goals"] = joint_goals
    env_A = MA4Rooms(**env_kwargs)

    env_A = MA4RoomsWrapper(env_A)

    Q_A, stats_A = Goal_Oriented_Q_learning(env_A, print_prefix="A", **hyper_params)

    library.save_extended_Q(Q_A, base_pkl_path + "Q_A.pkl")

    benchmark.follow_policies(env_A, Q_A, all_test_joint_starts, "trajs/Q_A_traj.txt")

    # -------- #
    #  Task B  #
    # -------- #
    # Agent 1 can go to any goal
    # Agent 2 must go to a right goal (i.e., TRC or BRC)
    g_right = [MA4Rooms.CORRIDOR_TRC, MA4Rooms.CORRIDOR_BRC]
    joint_goals = list(itertools.product(g_all, g_right))
    joint_goals = [list(el) for el in joint_goals]

    env_kwargs["joint_goals"] = joint_goals
    env_B = MA4Rooms(**env_kwargs)

    env_B = MA4RoomsWrapper(env_B)

    Q_B, stats_B = Goal_Oriented_Q_learning(env_B, print_prefix="B", **hyper_params)
    library.save_extended_Q(Q_B, base_pkl_path + "Q_B.pkl")

    benchmark.follow_policies(env_B, Q_B, all_test_joint_starts, "trajs/Q_B_traj.txt")

    test_composition(Q_A, Q_B, env_A)


def test_composition(Q_A, Q_B, env):
    # [(2, 2), (5, 2)]
    all_test_starts = [MA4Rooms.TL_CNR, MA4Rooms.TR_CNR, MA4Rooms.BL_CNR, MA4Rooms.BR_CNR]
    all_test_joint_starts = list(itertools.product(all_test_starts, all_test_starts))
    all_test_joint_starts = [list(el) for el in all_test_joint_starts]
    all_test_joint_starts.append([(2, 2), (5, 2)])
    # Remove joint_starts with collisions
    all_test_joint_starts = list(filter(lambda x: x[0] != x[1], all_test_joint_starts))

    # -------------------------- #
    #  Task (NOT A) AND (NOT B)  #
    # -------------------------- #
    # Agent 1 must go to a top goal (i.e., TLC, TRC)
    # Agent 2 must go to a left goal (i.e., TLC, BLC)

    n_actions = env.action_space.n

    AND_PTL = partial(AND, n_actions=n_actions)
    NOT_PTL = partial(NOT, n_actions=n_actions, rmin=-100, rmax=env.rmax)

    print("# A AND B #")
    Q_comp_and = AND_PTL(Q_A, Q_B)

    # env shouldn't really matter here since dynamics are the same and only the rewards are different
    benchmark.follow_policies(env, Q_comp_and, all_test_joint_starts, "trajs/Q_comp_and_traj.txt")

    Q_comp = NOT_PTL(AND_PTL(Q_A, Q_B))
    # # env shouldn't really matter here since dynamics are the same and only the rewards are different
    benchmark.follow_policies(env, Q_comp, all_test_joint_starts, "trajs/Q_comp_traj.txt")
    pass


def test_fn(x):
    return x * 2


def test_multiproc():
    with multiprocessing.Pool(processes=16) as pool:
        results = pool.map(test_fn, range(16))

    print(results)


def test_load():
    env_kwargs = {
        "n_agents": 2,
        "n_actions": 5,
        "goal_reward": 2,
        "collide_reward": -0.1,
        "joint_start_state": [(1, 1), (11, 11)],  # It currently doesn't work if this isn't specified
        "random_starts": True,
        "rooms_type": "corridors"
    }

    g_all = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC,
             MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_TRC]

    g_bottom = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC]

    joint_goals = list(itertools.product(g_bottom, g_all))
    joint_goals = [list(el) for el in joint_goals]

    env_kwargs["joint_goals"] = joint_goals
    env_A = MA4Rooms(**env_kwargs)

    env_A = MA4RoomsWrapper(env_A)

    Q_A = library.load_extended_Q("Q_pkls/corridors/3/Q_A.pkl", env_A.action_space.n)
    Q_B = library.load_extended_Q("Q_pkls/corridors/3/Q_B.pkl", env_A.action_space.n)

    test_composition(Q_A, Q_B, env_A)


if __name__ == '__main__':
    # main()
    test_load()
