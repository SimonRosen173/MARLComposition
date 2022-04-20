import pickle

from GridWorld import *
from library import Goal_Oriented_Q_learning, Q_learning, follow_extended_q_policy, follow_q_policy


def main():
    # I.e. both agents go left
    joint_goals = [[MA4Rooms.TLC, MA4Rooms.BLC], [MA4Rooms.BLC, MA4Rooms.TLC],
                   [MA4Rooms.TLC, MA4Rooms.TLC], [MA4Rooms.BLC, MA4Rooms.BLC]]
    joint_start_state = [(1, 1), (11, 11)]

    env = MA4Rooms(n_agents=2, n_actions=5,
                   goal_reward=2, collide_reward=-0.02,
                   joint_goals=joint_goals, joint_start_state=joint_start_state, random_starts=True)

    env = MA4RoomsWrapper(env)

    start_state = env.reset()

    print(start_state)
    env.render()

    # Q, stats = Q_learning(env, maxiter=200, is_printing=True, epsilon=0.2)
    # follow_q_policy(env, Q, is_rendering=True, render_mode="ascii")

    # maxiter = 100000
    maxiter = 1

    Q, stats = Goal_Oriented_Q_learning(env, maxiter=maxiter, is_printing=True, epsilon=0.2)

    follow_extended_q_policy(env, Q, is_rendering=True, render_mode="ascii")
    # print(Q)
    # print(stats)
    # Q, stats = Goal_Oriented_Q_learning(env)


if __name__ == '__main__':
    main()
