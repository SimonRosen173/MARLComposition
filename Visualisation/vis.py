from graphics import *
from PIL import Image
import os
import time
from typing import List, Tuple, Optional, Dict, Set
import imageio
import re

# class Vis:
#     pass


class VisGrid:
    def __init__(self, grid, window_dim, border_width=25, line_width=1, tick_time=0.1,
                 agent_color="blue", path_width=3, text_size=20, joint_goals=None, joint_start=None):
        self.grid = grid
        self.win_width = window_dim[0]
        self.win_height = window_dim[1]
        self.border_width = border_width  # E.g. offset before visualisations

        self.grid_width = len(grid[0])
        self.grid_height = len(grid)
        self.min_grid_dim = min(self.grid_width, self.grid_height)

        self.min_window_dim = min(self.win_height, self.win_width)
        self.min_usable_dim = self.min_window_dim - 2*border_width

        self.tile_size = int(self.min_usable_dim/self.min_grid_dim)

        self.window = GraphWin(width=self.win_width, height=self.win_height)

        self.line_width = line_width
        self.path_width = path_width
        self.circle_radius = (self.tile_size - 0.1*self.tile_size)/2
        self.text_size = text_size

        self.tick_time = tick_time
        self.agent_color = agent_color

        self._draw_grid()
        self._circles = None
        self._labels = None
        self._joint_goal_shapes = None

        self._agent_colors = ["red", "green", "blue", "orange", "DarkMagenta", "DarkRed"]  # "yellow", "cyan"]

        if joint_goals is not None:
            self._joint_goals = joint_goals
            self._draw_multi_goals(joint_goals)

        if joint_start is not None:
            self._joint_start = joint_start
            self._joint_state = joint_start

    def _draw_grid(self):
        win = self.window
        grid = self.grid

        start_x, start_y = self.border_width, self.border_width

        # Dimensions of grid in terms of pixels
        grid_width_x = self.grid_width*self.tile_size
        grid_height_y = self.grid_height * self.tile_size

        end_x = start_x + grid_width_x
        end_y = start_y + grid_height_y

        # Horizontal lines
        for y in range(start_y, end_y + self.tile_size, self.tile_size):
            line = Line(Point(start_x, y), Point(end_x, y))
            line.draw(win)

        # Vertical lines
        for x in range(start_x, end_x + self.tile_size, self.tile_size):
            line = Line(Point(x, start_y), Point(x, end_y))
            line.draw(win)

        # Obstacles
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                if grid[gy][gx] > 0:
                    x = self.border_width + gx * self.tile_size
                    y = self.border_width + gy * self.tile_size
                    rect = Rectangle(Point(x, y), Point(x+self.tile_size, y+self.tile_size))
                    rect.setFill("black")
                    rect.draw(win)

    def _draw_multi_goals(self, joint_goals):
        agent_colors = self._agent_colors

        # Draw goals -> Only shows goals for 2 agents rn
        self._joint_goal_shapes = []

        goal_margin = int(self.tile_size * 0.05)
        goal_shape_size = int(self.tile_size * 0.35)

        for joint_goal in joint_goals:
            # Only drawing for first 2 agents
            # Coord for agent 1 goal
            x1, y1 = self.get_coord_from_grid(joint_goal[0][1], joint_goal[0][0])
            # Coord for agent 2 goal
            x2, y2 = self.get_coord_from_grid(joint_goal[1][1], joint_goal[1][0])

            # Top left in tile
            rect_1 = Rectangle(Point(x1 + goal_margin,
                                     y1 + goal_margin),
                               Point(x1 + goal_margin + goal_shape_size,
                                     y1 + goal_margin + goal_shape_size))

            # Bottom right in tile
            rect_2 = Rectangle(Point(x2 + self.tile_size - (goal_shape_size + goal_margin),
                                     y2 + self.tile_size - (goal_shape_size + goal_margin)),
                               Point(x2 + self.tile_size - goal_margin,
                                     y2 + self.tile_size - goal_margin))

            rect_1.setOutline(agent_colors[0])
            rect_2.setOutline(agent_colors[1])

            rect_1.setFill(agent_colors[0])
            rect_2.setFill(agent_colors[1])

            rect_1.draw(self.window)
            rect_2.draw(self.window)

            self._joint_goal_shapes.append([rect_1, rect_2])
        pass

    def update_2_agents_pos(self, states):
        circle_radius = self.circle_radius

        # circle = Circle(Point(-1, -1), circle_radius)
        # agent_colors = ["red", "green", "blue", "orange", "DarkMagenta", "DarkRed"]  # "yellow", "cyan"]
        agent_colors = self._agent_colors[:2]
        win = self.window

        if self._circles is None:
            self._circles = []

            circles = []
            for i in range(2):
                pos = states[i]
                x, y = self.get_coord_from_grid(pos[1], pos[0])
                x = x + self.tile_size/2
                y = y + self.tile_size/2

                circles.append(Circle(Point(x, y), circle_radius))

            # circles = [Circle(Point(-1, -1), circle_radius) for _ in range(2)]

            for i, circle in enumerate(circles):
                curr_color = agent_colors[i % len(agent_colors)]
                circle.setOutline(curr_color)
                circle.setFill(curr_color)
                circle.draw(win)

                self._circles.append(circle)

        circles = self._circles
        for agent_ind in range(2):
            circle = circles[agent_ind]
            pos = states[agent_ind]
            x, y = self.get_coord_from_grid(pos[1], pos[0])
            x = x + self.tile_size/2
            y = y + self.tile_size/2

            curr_cent = circle.getCenter()
            curr_x, curr_y = int(curr_cent.x), int(curr_cent.y)
            VisGrid.move_to(circle, curr_x, curr_y, x, y)

        # for agent_ind in range(len(paths)):
        #     if point_ind >= len(paths[agent_ind]):
        #         continue
        #     circle = circles[agent_ind]
        #
        #     pos = paths[agent_ind][point_ind]
        #     if is_pos_xy:
        #         x, y = self.get_coord_from_grid(pos[0], pos[1])
        #     else:
        #         x, y = self.get_coord_from_grid(pos[1], pos[0])
        #     x = x + self.tile_size/2
        #     y = y + self.tile_size/2
        #
        #     curr_cent = circle.getCenter()
        #     curr_x, curr_y = curr_cent.x, curr_cent.y
        #     VisGrid.move_to(circle, curr_x, curr_y, x, y)


    def save_win_to_gif(self, file_name):
        # saves the current TKinter object in postscript format
        self.window.postscript(file="image.eps", colormode='color')

        # Convert from eps format to gif format using PIL
        img = Image.open("image.eps")
        img.save(file_name, "gif")

    def get_coord_from_grid(self, gx, gy):
        x = self.border_width + gx*self.tile_size
        y = self.border_width + gy * self.tile_size
        return x, y

    @staticmethod
    def move_to(obj, curr_x, curr_y, next_x, next_y):
        step_x = next_x - curr_x
        step_y = next_y - curr_y
        obj.move(step_x, step_y)

    # Path is a list of (x, y) tuples
    def animate_path(self, path, is_pos_xy=True):
        win = self.window
        circle_radius = self.circle_radius
        circle = Circle(Point(-1, -1), circle_radius)
        circle.setOutline(self.agent_color)
        circle.setFill(self.agent_color)

        circle.draw(win)
        for pos in path:
            if is_pos_xy:
                x, y = self.get_coord_from_grid(pos[0], pos[1])
            else:
                x, y = self.get_coord_from_grid(pos[1], pos[0])
            x = x + self.tile_size/2
            y = y + self.tile_size/2

            curr_cent = circle.getCenter()
            curr_x, curr_y = curr_cent.x, curr_cent.y
            VisGrid.move_to(circle, curr_x, curr_y, x, y)
            time.sleep(self.tick_time)

    def animate_multi_path(self, paths, is_pos_xy=True, joint_goals=None,
                           save_video=False, tmp_folder="", video_path=""):

        if joint_goals is None:
            joint_goals = []
        if len(joint_goals) > 0 and is_pos_xy:
            raise NotImplementedError("Goals must be specified as (y, x)")

        img_index = 0
        if save_video:
            # delete tmp images
            for f in os.listdir(tmp_folder):
                if re.search(r"img_\d+.png", str(f)) is not None:
                    os.remove(os.path.join(tmp_folder, f))

        win = self.window
        message = Text(Point(20, 10), "0")
        message.draw(win)

        circle_radius = self.circle_radius
        circles = [Circle(Point(-1, -1), circle_radius) for _ in range(len(paths))]
        # circle = Circle(Point(-1, -1), circle_radius)
        # agent_colors = ["red", "green", "blue", "orange", "DarkMagenta", "DarkRed"]  # "yellow", "cyan"]
        agent_colors = self._agent_colors
        # circle.setOutline(self.agent_color)
        # circle.setFill(self.agent_color)

        if self._circles is not None:
            for circle in self._circles:
                circle.undraw()
                del circle

        if self._labels is not None:
            for label in self._labels:
                label.undraw()
                del label

        if self._joint_goal_shapes is not None:
            for joint_goal_shape in self._joint_goal_shapes:
                for goal_shape in joint_goal_shape:
                    goal_shape.undraw()
                    del goal_shape

        self._draw_multi_goals(joint_goals)
        # # Draw goals -> Only shows goals for 2 agents rn
        # self._joint_goal_shapes = []
        #
        # goal_margin = int(self.tile_size * 0.05)
        # goal_shape_size = int(self.tile_size * 0.35)
        #
        # for joint_goal in joint_goals:
        #     # Only drawing for first 2 agents
        #     # Coord for agent 1 goal
        #     x1, y1 = self.get_coord_from_grid(joint_goal[0][1], joint_goal[0][0])
        #     # Coord for agent 2 goal
        #     x2, y2 = self.get_coord_from_grid(joint_goal[1][1], joint_goal[1][0])
        #
        #     # Top left in tile
        #     rect_1 = Rectangle(Point(x1 + goal_margin,
        #                              y1 + goal_margin),
        #                        Point(x1 + goal_margin + goal_shape_size,
        #                              y1 + goal_margin + goal_shape_size))
        #
        #     # Bottom right in tile
        #     rect_2 = Rectangle(Point(x2 + self.tile_size - (goal_shape_size + goal_margin),
        #                              y2 + self.tile_size - (goal_shape_size + goal_margin)),
        #                        Point(x2 + self.tile_size - goal_margin,
        #                              y2 + self.tile_size - goal_margin))
        #
        #     rect_1.setOutline(agent_colors[0])
        #     rect_2.setOutline(agent_colors[1])
        #
        #     rect_1.setFill(agent_colors[0])
        #     rect_2.setFill(agent_colors[1])
        #
        #     rect_1.draw(win)
        #     rect_2.draw(win)
        #
        #     self._joint_goal_shapes.append([rect_1, rect_2])

        self._circles = []
        for i, circle in enumerate(circles):
            curr_color = agent_colors[i % len(agent_colors)]
            circle.setOutline(curr_color)
            circle.setFill(curr_color)
            circle.draw(win)

            self._circles.append(circle)

        max_len = max([len(path) for path in paths])
        for point_ind in range(max_len):
            message.setText(f"{point_ind}")
            for agent_ind in range(len(paths)):
                if point_ind >= len(paths[agent_ind]):
                    continue
                circle = circles[agent_ind]

                pos = paths[agent_ind][point_ind]
                if is_pos_xy:
                    x, y = self.get_coord_from_grid(pos[0], pos[1])
                else:
                    x, y = self.get_coord_from_grid(pos[1], pos[0])
                x = x + self.tile_size/2
                y = y + self.tile_size/2

                curr_cent = circle.getCenter()
                curr_x, curr_y = curr_cent.x, curr_cent.y
                VisGrid.move_to(circle, curr_x, curr_y, x, y)

            if save_video:
                self.save_to_png(tmp_folder + f"/img_{img_index}")
                img_index += 1
            time.sleep(self.tick_time)

        if save_video:
            images = []
            for i in range(img_index):
                curr_file_path = f"{tmp_folder}/img_{i}.png"
                images.append(imageio.imread(curr_file_path))

            imageio.mimsave(video_path, images, format="gif", duration=0.5)

            # delete tmp images
            for f in os.listdir(tmp_folder):
                if re.search(r"img_\d+.png", str(f)) is not None:
                    os.remove(os.path.join(tmp_folder, f))

    def animate_mapd(self, agents, start_locs, is_pos_xy=True):
        win = self.window
        message = Text(Point(20, 10), "0")
        message.draw(win)

        circle_radius = self.circle_radius
        start_loc_coords = []
        if is_pos_xy:
            for pos in start_locs:
                x, y = self.get_coord_from_grid(pos[0], pos[1])
                start_loc_coords.append((x+circle_radius, y+circle_radius))
        else:
            for pos in start_locs:
                x, y = self.get_coord_from_grid(pos[1], pos[0])
                start_loc_coords.append((x+circle_radius, y+circle_radius))

        circles = [Circle(Point(coord[0], coord[1]), circle_radius) for coord in start_loc_coords]
        # agent_colors = ["red", "green", "blue", "yellow", "orange", "cyan"]
        agent_colors = ["red", "green", "blue", "orange", "DarkMagenta", "DarkRed"]  # "yellow", "cyan"]

        for i, circle in enumerate(circles):
            curr_color = agent_colors[i % len(agent_colors)]
            circle.setOutline(curr_color)
            circle.setFill(curr_color)
            circle.draw(win)

        # from MAPD.TokenPassing import Agent
        full_path_dicts = {}
        for i, agent in enumerate(agents):
            full_path = agent.get_full_path()
            if len(full_path) == 0:
                full_path_dicts[agent.id] = {0: start_loc_coords[i]}
            else:
                full_path_dicts[agent.id] = {tup[1]: tup[0] for tup in full_path}


        task_path_dicts = {}
        for agent in agents:
            task_hist = agent.task_history

        agent_ids = [agent.id for agent in agents]

        all_t = [list(full_path_dicts[agent.id].keys()) for agent in agents]
        max_t = -1
        for i in range(len(all_t)):
            if len(all_t[i]) > 0:
                max_t = max(max_t, max(all_t[i]))
        # max_t = max([max(subarr) for subarr in all_t])

        for t in range(max_t + 1):
            message.setText(f"{t}")
            for agent_id in agent_ids:
                curr_path = full_path_dicts[agent_id]
                if t in curr_path:
                    circle = circles[agent_id]
                    pos = curr_path[t]

                    if is_pos_xy:
                        x, y = self.get_coord_from_grid(pos[0], pos[1])
                    else:
                        x, y = self.get_coord_from_grid(pos[1], pos[0])
                    x = x + self.tile_size/2
                    y = y + self.tile_size/2

                    curr_cent = circle.getCenter()
                    curr_x, curr_y = curr_cent.x, curr_cent.y
                    VisGrid.move_to(circle, curr_x, curr_y, x, y)

            time.sleep(self.tick_time)
        # all_paths_le = [full_path_dicts[key] for key in full_path_dicts.keys()]
        # max_t = max([max([]) for path in all_paths])

        # max_t = max([max([tup[1] for tup in full_paths_dict[agent_ind]]) for agent_ind in full_paths_dict.keys()])
        # max_t = max([[max(full_path_dicts[agent.id].keys())] for agent in agents])

        print(max_t)
        pass

    def draw_path(self, path, all_arrows=False):
        win = self.window
        last_x, last_y = self.get_coord_from_grid(path[0][0], path[0][1])
        last_x = last_x + self.tile_size * 0.5
        last_y = last_y + self.tile_size * 0.5

        for i in range(1, len(path)):
            pos = path[i]
            curr_x, curr_y = self.get_coord_from_grid(pos[0], pos[1])
            curr_x = curr_x + self.tile_size * 0.5
            curr_y = curr_y + self.tile_size * 0.5

            line = Line(Point(last_x, last_y), Point(curr_x, curr_y))
            line.setWidth(self.path_width)
            if i == len(path) - 1:
                line.setArrow("last")
            elif all_arrows:
                line.setArrow("last")

            line.setFill(self.agent_color)
            line.draw(win)

            last_x, last_y = curr_x, curr_y

    def draw_start(self, pos):
        win = self.window
        x, y = self.get_coord_from_grid(pos[0], pos[1])
        x = x + self.tile_size*0.5
        y = y + self.tile_size * 0.5

        text = Text(Point(x, y), "S")
        text.setSize(self.text_size)
        text.draw(win)

    def draw_goal(self, pos):
        win = self.window
        x, y = self.get_coord_from_grid(pos[0], pos[1])
        x = x + self.tile_size * 0.5
        y = y + self.tile_size * 0.5

        text = Text(Point(x, y), "G")
        text.setSize(self.text_size)
        text.draw(win)

    def save_to_png(self, file_name):
        # save postscipt image
        self.window.postscript(file=file_name + '.eps')
        # use PIL to convert to PNG
        img = Image.open(file_name + '.eps')
        img.save(file_name + '.png', 'png')
        del img
        os.remove(file_name + '.eps')


class VisGraph:
    pass


def example():
    grid = [[0]*10 for i in range(10)]
    grid[1][1] = 1
    grid[1][2] = 1
    grid[1][3] = 1
    path = [(0, 0), (0, 1), (0, 2), (0, 3)]

    new_vis = VisGrid(grid, (400, 400), 25, tick_time=0.5)

    new_vis.draw_start(path[0])
    new_vis.draw_goal(path[-1])

    new_vis.window.getMouse()
    new_vis.animate_path(path)

    # new_vis.draw_path(path)
    # new_vis.save_to_png("test")

    new_vis.window.getMouse()
    new_vis.window.close()
    # print(grid)


def test_vis_grid():
    import numpy as np
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

    grid = [[int(el) for el in row.split(" ")] for row in grid_str.split("\n")]
    grid = np.asarray(grid)

    vis_grid = VisGrid(grid, (400, 400), 25, tick_time=0.5)
    vis_grid.window.getMouse()

    vis_grid.window.close()


if __name__ == "__main__":
    # example()
    test_vis_grid()
    # win = GraphWin(width=350, height=350)
    # Point(100, 50).draw(win)
    # print(win.winfo_width())
    # win.getMouse()
    # save_to_png(win, "imgs/img")
    # win.close()
    # print("")
