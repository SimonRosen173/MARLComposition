from GridWorld import *
from library import Goal_Oriented_Q_learning, Q_learning


def main():
    # I.e. both agents go left
    joint_goals = [[MA4Rooms.TLC, MA4Rooms.BLC], [MA4Rooms.BLC, MA4Rooms.TLC],
                   [MA4Rooms.TLC, MA4Rooms.TLC], [MA4Rooms.BLC, MA4Rooms.BLC]]
    joint_start_state = [(1, 1), (11, 11)]

    env = MA4Rooms(n_agents=2, n_actions=5,
                   goal_reward=10,
                   joint_goals=joint_goals, joint_start_state=joint_start_state, random_starts=True)

    env = MA4RoomsWrapper(env)

    start_state = env.reset()

    print(start_state)
    env.render()

    # Q, stats = Q_learning(env, maxiter=200, is_printing=True, epsilon=0.2)
    Q, stats = Goal_Oriented_Q_learning(env, maxiter=200, is_printing=True, epsilon=0.2)

    print(Q)
    print(stats)
    # Q, stats = Goal_Oriented_Q_learning(env)


if __name__ == '__main__':
    main()
