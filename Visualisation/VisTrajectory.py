from Visualisation import vis
from GridWorld import *
import os


def animate_traj_file(file_name, grid, joint_goals=None, tick_time=0.5,
                      save_video=False, tmp_folder="", video_path=""):
    traj_str_arr = []
    with open(file_name) as f:
        line_str = f.readline()
        # traj_str_arr.append(line_str)
        while line_str != "":
            traj_str_arr.append(line_str)
            line_str = f.readline()

    traj_arr = [[[(int(state.split(",")[0]), int(state.split(",")[1])) for state in joint_state.split(";")]
                 for joint_state in traj_str.replace("\n", "").split("|")]
                for traj_str in traj_str_arr]

    paths = [[list(el) for el in list(zip(*traj))] for traj in traj_arr]

    vis_grid = vis.VisGrid(grid, (400, 400), 25, tick_time=tick_time)
    if not save_video:
        vis_grid.window.getMouse()

    for i in range(len(paths)):
        vis_grid.animate_multi_path(paths[i], is_pos_xy=False, joint_goals=joint_goals,
                                    save_video=save_video, tmp_folder=tmp_folder,
                                    video_path=video_path + f"/vid_{i}.gif")
        if not save_video:
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


def vis_corridors(folder_no, file_name, task_type, save_video=False, tick_time=0.5):
    g_all = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC,
             MA4Rooms.CORRIDOR_TLC, MA4Rooms.CORRIDOR_TRC]
    g_bottom = [MA4Rooms.CORRIDOR_BLC, MA4Rooms.CORRIDOR_BRC]
    g_right = [MA4Rooms.CORRIDOR_TRC, MA4Rooms.CORRIDOR_BRC]
    joint_goals = None

    if task_type == "A" or task_type == "A_SNG":
        # Agent 1 must go to a bottom goal (i.e., BLC or BRC)
        # Agent 2 can go to any goal
        joint_goals = list(itertools.product(g_bottom, g_all))

    elif task_type == "B" or task_type == "B_SNG":
        # Agent 1 can go to any goal
        # Agent 2 must go to a right goal (i.e., TRC or BRC)
        joint_goals = list(itertools.product(g_all, g_right))
    elif task_type == "A_AND_B" or task_type == "A_AND_B_SNG":
        # Agent 1 must go to a bottom goal (i.e., BLC or BRC)
        # Agent 2 must go to a right goal (i.e., TRC or BRC)
        joint_goals = list(itertools.product(g_bottom, g_right))

    if joint_goals is not None:
        joint_goals = [list(el) for el in joint_goals]

    joint_start_state = [(1, 1), (11, 11)]

    env = MA4Rooms(n_agents=2, n_actions=5,
                   joint_goals=joint_goals, joint_start_state=joint_start_state,
                   random_starts=True, rooms_type="corridors")
    grid = env.get_grid()

    # agent 1 -> red
    # agent 2 -> green

    traj_path = full_path + f"/trajs/corridors/{folder_no}/{file_name}"
    tmp_folder_path = full_path + "/traj_gifs/tmp"
    video_path = full_path + f"/traj_gifs/{folder_no}/{task_type}"
    animate_traj_file(traj_path, grid, joint_goals=joint_goals,
                      tick_time=tick_time,
                      save_video=save_video, tmp_folder=tmp_folder_path, video_path=video_path)


if __name__ == "__main__":
    vis_corridors(20, "Q_A_and_B_sng_traj.txt", task_type="A_AND_B_SNG", save_video=True, tick_time=0.01)

