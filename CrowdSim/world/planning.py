import heapq
import numpy as np
import cv2
from scipy.interpolate import splprep, splev

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# 定义八个方向（包括斜向）
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] # 斜向


class Path_Planner():
    def __init__(
        self,
        map_free,
        map_resolution: float,
        step_size_m: float = 0.5,
        clearance_m: float = 0.2,
        smooth=False,
        viz=False,
        verbose: bool = True,
    ):
        self.map_resolution = float(map_resolution)
        self.step_size_m = float(step_size_m)
        self.clearance_m = float(clearance_m)
        self.step_px = max(1, int(round(self.step_size_m / self.map_resolution)))
        # ``clearance_m`` is a radius around occupied cells, not a kernel
        # width.  The previous square kernel of clearance/res pixels provided
        # only about half the requested clearance on each side.
        self.dilate_px = max(1, int(np.ceil(self.clearance_m / self.map_resolution)))
        self.smooth = smooth
        self.map_dialate = self.post_proc_map(map_free)
        self.map_size = self.map_dialate.shape
        self.directions = self._build_directions(self.step_px)
        # A* checks the same short line segments millions of times. Cache the
        # integer raster offsets for each displacement instead of allocating a
        # new linspace array in every edge check (Office was spending most of
        # its planning time here).
        self._edge_offset_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        # Corridor candidates often converge to the same navigable pixel
        # pairs. Keep a small bounded cache so fallback modes do not rerun the
        # same A* search while avoiding unbounded memory use in long jobs.
        self._astar_path_cache: dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray | None] = {}

        self.viz = viz
        self.verbose = verbose

    def post_proc_map(self, map):
        diameter = 2 * self.dilate_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (diameter, diameter)
        )
        dilated_img = cv2.dilate(map, kernel, iterations=1)

        return dilated_img

    def get_astar_path(self, start, goal):

        start = tuple(start)
        goal = tuple(goal)

        cache_key = (start, goal)
        if cache_key in self._astar_path_cache:
            cached = self._astar_path_cache[cache_key]
            return None if cached is None else cached.copy()

        g_score, parents = self.get_cost(start, goal)

        if g_score[goal] != float('inf'):
            path = self.reconstruct_path(parents, start, goal)
            # control_points = self.select_control_points(path)
            control_points = path

            if self.smooth:

                # 使用B样条平滑控制点路径
                smoothed_path = self.smooth_with_b_spline(control_points, path.shape[0])
                print('[Astar] Found b-spline path with: ', smoothed_path.shape)

                # 可视化结果
                if self.viz:

                    self.viz_cost_and_path(g_score, start, goal, smoothed_path)
                    # plt.scatter(control_points[:,1], control_points[:,0], color='g')

                result = smoothed_path
            else:
                result = control_points
        else:
            if self.verbose:
                print("[Astar] !!! No path found from start to goal.")
            if self.viz:
                self.viz_cost_and_path(g_score, start, goal, path=None)
            result = None

        if len(self._astar_path_cache) >= 512:
            self._astar_path_cache.pop(next(iter(self._astar_path_cache)))
        self._astar_path_cache[cache_key] = (
            None if result is None else np.asarray(result).copy()
        )
        return None if result is None else np.asarray(result).copy()



    def get_cost(self, start, goal):
        """A* search returning (g_score array, parent array).

        Stores true g-scores (cost from start) in the distance array and uses
        f = g + h as the priority-queue key.  An early-exit shortcut fires when
        the current node is within one step of the goal and has a clear edge.
        """
        u_lim, v_lim = self.map_size

        # Pre-compute direction weights once (avoid np.linalg.norm in the hot loop).
        dir_weights = [float(np.linalg.norm(d)) for d in self.directions]

        def _h(node: tuple) -> float:
            return float(np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2))

        g_score = np.full((u_lim, v_lim), np.inf)
        g_score[start] = 0.0
        parent = np.full((u_lim, v_lim, 2), -1)
        pq = [(_h(start), start)]          # (f = g + h, node)
        closed: set = set()

        while pq:
            _, current_node = heapq.heappop(pq)

            if current_node in closed:
                continue                    # stale queue entry — skip
            closed.add(current_node)

            if current_node == goal:
                break

            g_cur = g_score[current_node]

            # Early-exit: jump straight to goal if within one step and clear.
            h_cur = _h(current_node)
            if h_cur <= self.step_px and self._edge_is_free(current_node, goal):
                new_g = g_cur + h_cur
                if new_g < g_score[goal]:  # only improve; don't overwrite a better path
                    g_score[goal] = new_g
                    parent[goal] = current_node
                break

            for direction, w in zip(self.directions, dir_weights):
                neighbor = (current_node[0] + direction[0], current_node[1] + direction[1])
                if not (0 <= neighbor[0] < u_lim and 0 <= neighbor[1] < v_lim):
                    continue
                if self.map_dialate[neighbor] != 0:
                    continue
                if neighbor in closed:
                    continue
                if not self._edge_is_free(current_node, neighbor):
                    continue
                new_g = g_cur + w
                if new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    parent[neighbor] = current_node
                    heapq.heappush(pq, (new_g + _h(neighbor), neighbor))

        return g_score, parent

    @staticmethod
    def _build_directions(step_px):
        return [
            (dy * step_px, dx * step_px)
            for dy, dx in DIRECTIONS
        ]

    def _edge_is_free(self, start, goal):
        y0, x0 = int(start[0]), int(start[1])
        dy, dx = int(goal[0]) - y0, int(goal[1]) - x0
        if max(abs(dy), abs(dx)) <= 1:
            return True
        key = (dy, dx)
        offsets = self._edge_offset_cache.get(key)
        if offsets is None:
            steps = max(abs(dy), abs(dx))
            t = np.arange(steps + 1, dtype=np.float32) / float(steps)
            offsets = (
                np.rint(t * dy).astype(np.int64),
                np.rint(t * dx).astype(np.int64),
            )
            self._edge_offset_cache[key] = offsets
        rows, cols = offsets
        return bool(np.all(self.map_dialate[y0 + rows, x0 + cols] == 0))

    def reconstruct_path(self, parent, start, goal):

        # 从终点回溯到起点，重建最短路径
        x, y = goal
        path = []
        while parent[x, y][0] != -1:
            path.append((x, y))
            x, y = parent[x, y]
        path.append(start)
        path.reverse()
        return np.array(path)

    def calculate_curvature(self, path):
        # 获取x和y坐标
        x = path[:, 0]
        y = path[:, 1]

        # 计算三角形的面积部分 (x3 - x1)*(y2 - y1) - (y3 - y1)*(x2 - x1)
        dx1 = x[2:] - x[:-2]  # x3 - x1
        dy1 = y[2:] - y[:-2]  # y3 - y1
        dx2 = x[1:-1] - x[:-2]  # x2 - x1
        dy2 = y[1:-1] - y[:-2]  # y2 - y1

        # 计算曲率的分子部分
        area = np.abs(dx1 * dy2 - dy1 * dx2)

        # 计算路径长度的3/2次方部分 (x2 - x1)^2 + (y2 - y1)^2
        length_sq = (dx2 ** 2 + dy2 ** 2) ** 1.5

        # 计算曲率，避免除零错误
        curvatures = np.divide(2 * area, length_sq, where=length_sq != 0)

        return curvatures

    def select_control_points(self, path, curvature_threshold=0.2):
        curvatures = self.calculate_curvature(path)
        control_points = []  # 始终包括起点 path[0]
        j = 0
        for i, curv in enumerate(curvatures):
            if curv > curvature_threshold:
                control_points.append(path[i + 1])  # 曲率变化较大的点作为控制点
                j = 0
            else:
                j += 1
                if j > 4:
                    control_points.append(path[i])
                    j = 0
        control_points.append(path[-1])
        return np.array(control_points)


    def smooth_with_b_spline(self, control_points, num):

        tck, u = splprep(control_points.T, s=0)  # 使用B样条拟合
        x = np.linspace(0, 1, num)
        new_points = splev(x, tck)  # 插值100个点
        return np.array(new_points).T


    def viz_cost_and_path(self, distance, start, goal, path):
        plt.figure()
        plt.imshow(self.map_dialate)
        plt.imshow(distance)
        plt.plot(start[1], start[0],'gx')
        plt.plot(goal[1], goal[0], 'rx')
        if path is not None:
            plt.plot( path[:, 1], path[:, 0], color='b')
        # plt.show()
