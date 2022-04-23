from Visualisation import vis
from GridWorld import *
import os


def animate_traj_file(file_name, grid, goals=None):
    traj_str_arr = []
    with open(file_name) as f:
        line_str = f.readline()
        # traj_str_arr.append(line_str)
        while line_str != "":
            traj_str_arr.append(line_str)
            line_str = f.readline()

    traj_arr = [[[(int(state.split(",")[0]), int(state.split(",")[1])) for state in joint_state.split(";")]
                 for joint_state in traj_str.replace("\n","").split("|")]
                for traj_str in traj_str_arr]

    paths = [[list(el) for el in list(zip(*traj))] for traj in traj_arr]

    vis_grid = vis.VisGrid(grid, (400, 400), 25, tick_time=0.5)
    vis_grid.window.getMouse()

    for i in range(len(paths)):
        vis_grid.animate_multi_path(paths[i], is_pos_xy=False)
        vis_grid.window.getMouse()

    vis_grid.window.close()


curr_dir = os.getcwd()
full_path = "\\".join(curr_dir.split("\\")[:-1])  # Remove last folder from path


def vis_A():
    joint_goals = [[MA4Rooms.TLC, MA4Rooms.BRC]]
    joint_start_state = [(1, 1), (11, 11)]
    env = MA4Rooms(n_agents=2, n_actions=5,
                   joint_goals=joint_goals, joint_start_state=joint_start_state, random_starts=True)
    grid = env.get_grid()

    # agent 1 -> red
    # agent 2 -> green
    # Agent 1 (red) must go to a bottom goal (i.e., BLC or BRC)
    # Agent 2 (green) can go to any goal
    animate_traj_file(full_path + "/trajs/5/Q_A_traj.txt", grid)


def vis_B():
    joint_goals = [[MA4Rooms.TLC, MA4Rooms.BRC]]
    joint_start_state = [(1, 1), (11, 11)]
    env = MA4Rooms(n_agents=2, n_actions=5,
                   joint_goals=joint_goals, joint_start_state=joint_start_state, random_starts=True)
    grid = env.get_grid()

    # agent 1 -> red
    # agent 2 -> green
    # Agent 1 can go to any goal
    # Agent 2 must go to a right goal (i.e., TRC or BRC)
    animate_traj_file(full_path + "/trajs/5/Q_B_traj.txt", grid)


# C = A AND B
def vis_C():
    joint_goals = [[MA4Rooms.TLC, MA4Rooms.BRC]]
    joint_start_state = [(1, 1), (11, 11)]
    env = MA4Rooms(n_agents=2, n_actions=5,
                   joint_goals=joint_goals, joint_start_state=joint_start_state, random_starts=True)
    grid = env.get_grid()

    # agent 1 -> red
    # agent 2 -> green
    # agent 1 (red) must go to a bottom goal
    # agent 2 (green) must go to a right goal
    animate_traj_file(full_path + "/trajs/5/Q_comp_and_traj.txt", grid)


def vis_corridors(folder_no, file_name):
    joint_goals = [[MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_BRC]]
    joint_start_state = [(1, 1), (11, 11)]
    env = MA4Rooms(n_agents=2, n_actions=5,
                   joint_goals=joint_goals, joint_start_state=joint_start_state,
                   random_starts=True, rooms_type="corridors")
    grid = env.get_grid()

    # agent 1 -> red
    # agent 2 -> green
    # Agent 1 (red) must go to a bottom goal (i.e., BLC or BRC)
    # Agent 2 (green) can go to any goal
    animate_traj_file(full_path + f"/trajs/corridors/{folder_no}/{file_name}", grid)
    # Q_A_traj.txt


if __name__ == "__main__":
    vis_corridors(9, "Q_comp_traj.txt")

