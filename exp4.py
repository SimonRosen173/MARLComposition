from functools import partial

from GridWorld import *
from library import Goal_Oriented_Q_learning, Q_learning, follow_extended_q_policy, follow_q_policy
import library
from library import OR, AND, NOT
import benchmark
import multiprocessing

import itertools
import os

FOLDER_NO = 4
TRAJ_BASE_PATH = f"trajs/corridors_alt/{FOLDER_NO}"
BASE_PKL_PATH = f"Q_pkls/corridors_alt/{FOLDER_NO}"
MAXITERS = 10000

grid_file_path = os.getcwd() + "\\maps\\corridors_alt.txt"

ENV_KWARGS = {
    "grid_file": grid_file_path,
    "n_agents": 2,
    "n_actions": 5,
    "goal_reward": 20,
    "wait_reward": -0.01,
    "step_reward": -0.02,
    "wait_at_goal_reward": -0.001,
    "terminal_reward": -20,
    "collide_reward": -0.1,
    "rmin": -2000,  # max_steps * collide_reward - 10
    "joint_start_state": [(1, 1), (11, 1)],  # It currently doesn't work if this isn't specified
    "random_starts": True,
    "rooms_type": "corridors",
    # "max_steps": 100
}

HYPER_PARAMS = {
    "maxiter": MAXITERS,

    "epsilon": 0.25,  # Will be ignored if is_eps_decay = True

    "is_printing": True,
    "print_interval": 10,

    "is_eps_decay": True,
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_decay_rate": 1.0 / MAXITERS
}


# def main():
#     base_pkl_path = BASE_PKL_PATH
#     # base_pkl_path = "Q_pkls/corridors/4/"
#
#     env_kwargs = ENV_KWARGS
#     maxiters = MAXITERS
#     hyper_params = HYPER_PARAMS
#
#     # env_kwargs = {
#     #     "n_agents": 2,
#     #     "n_actions": 5,
#     #     "goal_reward": 2,
#     #     "collide_reward": -0.1,
#     #     "joint_start_state": [(1, 1), (11, 11)],  # It currently doesn't work if this isn't specified
#     #     "random_starts": True,
#     #     "rooms_type": "corridors"
#     # }
#
#     # maxiters = 50000
#
#     all_test_starts = [MA4Rooms.TL_CNR, MA4Rooms.TR_CNR, MA4Rooms.BL_CNR, MA4Rooms.BR_CNR]
#     all_test_joint_starts = list(itertools.product(all_test_starts, all_test_starts))
#     all_test_joint_starts = [list(el) for el in all_test_joint_starts]
#     all_test_joint_starts.append([(2, 2), (5, 2)])
#     # Remove joint_starts with collisions
#     all_test_joint_starts = list(filter(lambda x: x[0] != x[1], all_test_joint_starts))
#
#     g_all = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC,
#              MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_TRC]
#
#     # -------- #
#     #  Task A  #
#     # -------- #
#     # Agent 1 must go to a bottom goal (i.e., BLC or BRC)
#     # Agent 2 can go to any goal
#     g_bottom = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC]
#
#     joint_goals = list(itertools.product(g_bottom, g_all))
#     joint_goals = [list(el) for el in joint_goals]
#
#     env_kwargs["joint_goals"] = joint_goals
#     env_A = MA4Rooms(**env_kwargs)
#
#     env_A = MA4RoomsWrapper(env_A)
#
#     Q_A, stats_A = Goal_Oriented_Q_learning(env_A, print_prefix="A", **hyper_params)
#
#     library.save_extended_Q(Q_A, base_pkl_path + "Q_A.pkl")
#
#     benchmark.follow_policies(env_A, Q_A, all_test_joint_starts,
#                               f"trajs/corridors/{TRAJ_FOLDER_NO}/Q_A_traj.txt")
#
#     # -------- #
#     #  Task B  #
#     # -------- #
#     # Agent 1 can go to any goal
#     # Agent 2 must go to a right goal (i.e., TRC or BRC)
#     g_right = [MA4Rooms.CORRIDOR_TRC, MA4Rooms.CORRIDOR_BRC]
#     joint_goals = list(itertools.product(g_all, g_right))
#     joint_goals = [list(el) for el in joint_goals]
#
#     env_kwargs["joint_goals"] = joint_goals
#     env_B = MA4Rooms(**env_kwargs)
#
#     env_B = MA4RoomsWrapper(env_B)
#
#     Q_B, stats_B = Goal_Oriented_Q_learning(env_B, print_prefix="B", **hyper_params)
#     library.save_extended_Q(Q_B, base_pkl_path + "Q_B.pkl")
#
#     benchmark.follow_policies(env_B, Q_B, all_test_joint_starts,
#                               f"trajs/corridors/{TRAJ_FOLDER_NO}/Q_B_traj.txt")
#
#     test_composition(Q_A, Q_B, env_A)


def test_composition(Q_A, Q_B, env):
    # [(2, 2), (5, 2)]
    all_test_starts = [MA4Rooms.TL_CNR, MA4Rooms.TR_CNR, MA4Rooms.BL_CNR]
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
    NOT_PTL = partial(NOT, n_actions=n_actions, rmin=env.rmin, rmax=env.rmax)

    print("# A AND B #")
    Q_comp_and = AND_PTL(Q_A, Q_B)

    # env shouldn't really matter here since dynamics are the same and only the rewards are different
    benchmark.follow_policies(env, Q_comp_and, all_test_joint_starts,
                              f"{TRAJ_BASE_PATH}/Q_comp_and_traj.txt")

    Q_comp = NOT_PTL(AND_PTL(Q_A, Q_B))
    # # env shouldn't really matter here since dynamics are the same and only the rewards are different
    benchmark.follow_policies(env, Q_comp, all_test_joint_starts, f"{TRAJ_BASE_PATH}/Q_comp_traj.txt")
    pass


def train_task(task_id):
    maxiters = MAXITERS
    env_kwargs = ENV_KWARGS
    hyper_params = HYPER_PARAMS

    # hyper_params = {
    #     "maxiter": maxiters,
    #
    #     "epsilon": 0.25,  # Will be ignored if is_eps_decay = True
    #
    #     "is_printing": True,
    #
    #     "is_eps_decay": True,
    #     "eps_start": 1.0,
    #     "eps_end": 0.01,
    #     "eps_decay_rate": 1.0 / maxiters
    # }

    all_test_starts = [MA4Rooms.TL_CNR, MA4Rooms.TR_CNR, MA4Rooms.BL_CNR]
    all_test_joint_starts = list(itertools.product(all_test_starts, all_test_starts))
    all_test_joint_starts = [list(el) for el in all_test_joint_starts]
    # Remove joint_starts with collisions
    all_test_joint_starts = list(filter(lambda x: x[0] != x[1], all_test_joint_starts))

    g_all = [MA4Rooms.CORRIDOR_BLC,
             MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_TRC]

    if task_id == "A":
        # -------- #
        #  Task A  #
        # -------- #
        # Agent 1 must go to BLC
        # Agent 2 can go to any goal
        g_bottom = [MA4Rooms.CORRIDOR_BLC]

        joint_goals = list(itertools.product(g_bottom, g_all))
        joint_goals = [list(el) for el in joint_goals]

        env_kwargs["joint_goals"] = joint_goals
        env_A = MA4Rooms(**env_kwargs)

        env_A = MA4RoomsWrapper(env_A)

        Q_A, stats_A = Goal_Oriented_Q_learning(env_A, print_prefix=task_id, **hyper_params)

        print(f"[A] - min return = {np.min(stats_A['R'])}")

        library.save_extended_Q(Q_A, f"{BASE_PKL_PATH}/Q_A.pkl")

        benchmark.follow_policies(env_A, Q_A, all_test_joint_starts, f"{TRAJ_BASE_PATH}/Q_A_traj.txt")

        # return Q_A, stats_A
    else:
        # -------- #
        #  Task B  #
        # -------- #
        # Agent 1 can go to any goal
        # Agent 2 must go to TRC
        g_right = [MA4Rooms.CORRIDOR_TRC]
        joint_goals = list(itertools.product(g_all, g_right))
        joint_goals = [list(el) for el in joint_goals]

        env_kwargs["joint_goals"] = joint_goals
        env_B = MA4Rooms(**env_kwargs)

        env_B = MA4RoomsWrapper(env_B)

        Q_B, stats_B = Goal_Oriented_Q_learning(env_B, print_prefix=task_id, **hyper_params)

        print(f"[B] - min return = {np.min(stats_B['R'])}")

        library.save_extended_Q(Q_B, f"{BASE_PKL_PATH}/Q_B.pkl")

        benchmark.follow_policies(env_B, Q_B, all_test_joint_starts, f"{TRAJ_BASE_PATH}/Q_B_traj.txt")

        # return Q_B, stats_B


def train_multiproc():
    # Create folders if necessary
    if not os.path.isdir(TRAJ_BASE_PATH):
        os.makedirs(TRAJ_BASE_PATH)
    if not os.path.isdir(BASE_PKL_PATH):
        os.makedirs(BASE_PKL_PATH)

    with multiprocessing.Pool(processes=2) as pool:
        pool.map(train_task, ["A", "B"])

    Q_A, Q_B, env = load_Q(BASE_PKL_PATH)
    test_composition(Q_A, Q_B, env)


def load_Q(base_path):
    env_kwargs = ENV_KWARGS
    g_all = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC,
             MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_TRC]

    g_bottom = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC]

    joint_goals = list(itertools.product(g_bottom, g_all))
    joint_goals = [list(el) for el in joint_goals]

    env_kwargs["joint_goals"] = joint_goals
    env = MA4Rooms(**env_kwargs)

    env = MA4RoomsWrapper(env)

    Q_A = library.load_extended_Q(base_path + "/Q_A.pkl")
    Q_B = library.load_extended_Q(base_path + "/Q_B.pkl")

    return Q_A, Q_B, env


def load_Q_env_AB(folder_no):
    env_kwargs = ENV_KWARGS
    hyper_params = HYPER_PARAMS

    g_all = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC,
             MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_TRC]

    ################
    # ENVIRONMENTS #
    ################

    # ENV A
    g_bottom = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC]

    joint_goals = list(itertools.product(g_bottom, g_all))
    joint_goals = [list(el) for el in joint_goals]

    env_kwargs["joint_goals"] = joint_goals
    env_A = MA4Rooms(**env_kwargs)
    env_A = MA4RoomsWrapper(env_A)

    # ENV B
    g_right = [MA4Rooms.CORRIDOR_TRC, MA4Rooms.CORRIDOR_BRC]
    joint_goals = list(itertools.product(g_all, g_right))
    joint_goals = [list(el) for el in joint_goals]

    env_kwargs["joint_goals"] = joint_goals
    env_B = MA4Rooms(**env_kwargs)
    env_B = MA4RoomsWrapper(env_B)

    ##########
    # LOAD Q #
    ##########
    Q_A = library.load_extended_Q(f"{TRAJ_BASE_PATH}/Q_A.pkl")
    Q_B = library.load_extended_Q(f"{TRAJ_BASE_PATH}/Q_B.pkl")

    return env_A, env_B, Q_A, Q_B


def test_load(folder_no):
    # Load envs and Qs
    env_A, env_B, Q_A, Q_B = load_Q_env_AB(folder_no)

    ###################
    # FOLLOW POLICIES #
    ###################
    all_test_starts = [MA4Rooms.TL_CNR, MA4Rooms.TR_CNR, MA4Rooms.BL_CNR, MA4Rooms.BR_CNR]
    all_test_joint_starts = list(itertools.product(all_test_starts, all_test_starts))
    all_test_joint_starts = [list(el) for el in all_test_joint_starts]
    # Remove joint_starts with collisions
    all_test_joint_starts = list(filter(lambda x: x[0] != x[1], all_test_joint_starts))

    benchmark.follow_policies(env_A, Q_A, all_test_joint_starts, f"{TRAJ_BASE_PATH}/Q_A_traj.txt")
    benchmark.follow_policies(env_B, Q_B, all_test_joint_starts, f"{TRAJ_BASE_PATH}/Q_B_traj.txt")

    ####################
    # TEST COMPOSITION #
    ####################
    test_composition(Q_A, Q_B, env_A)


def follow_and_policy(load_folder_no, joint_start_state):
    env_A, env_B, Q_A, Q_B = load_Q_env_AB(load_folder_no)

    # # Follow policy for A
    # library.follow_extended_q_policy(env_A, Q_A, joint_start_state=joint_start_state,
    #                                  is_rendering=True, render_mode="canvas", render_delay=0.5)
    #
    # traj_str = env_A.get_trajectory(is_delimited=True)
    #
    # with open(f"trajs/corridors/{TRAJ_FOLDER_NO}/Q_A_sng_traj.txt", "w") as f:
    #     f.write(traj_str + "\n")

    # Follow policy for B
    library.follow_extended_q_policy(env_B, Q_B, joint_start_state=joint_start_state,
                                     is_rendering=True, render_mode="canvas", render_delay=0.5)

    traj_str = env_B.get_trajectory(is_delimited=True)

    with open(f"{TRAJ_BASE_PATH}/Q_B_sng_traj.txt", "w") as f:
        f.write(traj_str + "\n")

    # ###########
    # # A AND B #
    # ###########
    #
    # n_actions = env_A.action_space.n
    #
    # AND_PTL = partial(AND, n_actions=n_actions)
    #
    # Q_comp_and = AND_PTL(Q_A, Q_B)

    # # ENV A AND B
    # env_kwargs = ENV_KWARGS
    # g_bottom = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC]
    # g_right = [MA4Rooms.CORRIDOR_TRC, MA4Rooms.CORRIDOR_BRC]
    #
    # joint_goals = list(itertools.product(g_bottom, g_right))
    # joint_goals = [list(el) for el in joint_goals]
    #
    # env_kwargs["joint_goals"] = joint_goals
    # env_A_and_B = MA4Rooms(**env_kwargs)
    # env_A_and_B = MA4RoomsWrapper(env_A_and_B)
    #
    # library.follow_extended_q_policy(env_A_and_B, Q_comp_and, joint_start_state=joint_start_state,
    #                                  is_rendering=True, render_mode="canvas", render_delay=0.5)
    #
    # traj_str = env_A_and_B.get_trajectory(is_delimited=True)
    #
    # with open(f"trajs/corridors/{TRAJ_FOLDER_NO}/Q_A_and_B_sng_traj.txt", "w") as f:
    #     f.write(traj_str + "\n")


def print_goals():
    g_all = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC,
             MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_TRC]

    # ENV A
    g_bottom = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC]

    joint_goals = list(itertools.product(g_bottom, g_all))
    joint_goals = [list(el) for el in joint_goals]
    print(joint_goals)


if __name__ == '__main__':
    # main()
    # test_load(17)
    train_multiproc()
    # follow_and_policy(21, [(3, 2), (4, 2)])

