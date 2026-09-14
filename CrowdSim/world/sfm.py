import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import cv2


class Social_Force:
    def __init__(self, edt, cfg):
        self.name = 'sfm'
        self.dt = cfg['env']['dt']
        self.rad = cfg['agent']['radius']
        self.max_speed = cfg['agent']['max_vel']      

        self.gain_k = 5
        self.gain_a_static = 15
        self.gain_b_static = 0.25

        self.gain_a_agent = 25
        self.gain_b_agent = 0.5
        self.gain_ttc = float(cfg['env'].get('ttc_gain', 12.0))
        
        self.distance_field = edt
        self.map_resolution = float(cfg['map'].get('resolution', 1.0))
        self.grad_map = self.compute_gradient(self.distance_field, self.map_resolution)

        self.safe_dis = cfg['env']['safe_distance']
        self.avoid_dis = float(cfg['env'].get('neighbor_radius', 4.0))
        self.reach_dis = float(cfg['env'].get('reach_distance', 2*self.rad))
        self.prediction_horizon = float(
            cfg['env'].get('prediction_horizon', 2.0)
        )
        self.ttc_threshold = float(cfg['env'].get('ttc_threshold', 1.5))

        self.fov = False
        

    @staticmethod
    def compute_gradient(distance_field, map_resolution):
        """计算EDT的梯度场"""
        grad_y, grad_x = np.gradient(distance_field, map_resolution, map_resolution)
        grad_world_x = grad_x
        grad_world_y = -grad_y

        # 计算梯度模长
        magnitude = np.sqrt(grad_world_x**2 + grad_world_y**2)

        # 处理零梯度区域
        magnitude[magnitude == 0] = 1e-6  # 避免除以零

        # 归一化梯度
        grad_norm = np.stack([grad_world_x / magnitude, grad_world_y / magnitude], axis=0)

        return grad_norm
    
    @staticmethod
    def fov_filter(curr_dir, rel_obs):
        near_nbrs_idx, nbrs_dis, nbrs_relpos, nbrs_relvel = rel_obs

        if len(near_nbrs_idx) == 0:
            nbrs_idx = []
            nbrs_dis = None
            nbrs_pos = None

            return nbrs_idx, nbrs_dis, nbrs_pos, nbrs_relvel

        cos_angles = np.dot(-nbrs_relpos, curr_dir)

        # 筛选在视野范围内的邻居（夹角在±90度内，即cos值>0）
        in_fov = cos_angles >= 0

        fov_nbrs_idx = near_nbrs_idx[in_fov]
        fov_nbrs_dis = nbrs_dis[in_fov]
        fov_nbrs_relpos = nbrs_relpos[in_fov]
        
        if len(fov_nbrs_idx) > 5:
            sorted_indices = np.argsort(fov_nbrs_dis)[:5]          
            nbrs_idx = fov_nbrs_idx[sorted_indices]
            nbrs_dis = fov_nbrs_dis[sorted_indices]
            nbrs_pos = fov_nbrs_relpos[sorted_indices]
            nbrs_vel = nbrs_relvel[sorted_indices]
        else:
            nbrs_idx = fov_nbrs_idx
            nbrs_dis = fov_nbrs_dis
            nbrs_pos = fov_nbrs_relpos
            nbrs_vel = nbrs_relvel

        return nbrs_idx, nbrs_dis, nbrs_pos, nbrs_vel


    def compute_forces(self, agent_state, rel_obs):
        pos, cord_int, vel, goal = agent_state
        nbrs_idx, nbrs_dis, nbrs_relpos, nbrs_relvel = (
            self.fov_filter(vel, rel_obs) if self.fov else rel_obs
        )
        pixel_y, pixel_x = cord_int
        
        # ===== 1. 计算目标吸引力 =====
        direction_to_goal = goal - pos
        distance_to_goal = np.linalg.norm(direction_to_goal)
        
        if distance_to_goal > self.reach_dis:
            attractive_force = self.max_speed * direction_to_goal/distance_to_goal
        else:
            # print('[SFM] !!! Reach')
            attractive_force = np.zeros(2)
        
        
        # ===== 2. 计算障碍排斥力 =====
        dist = self.distance_field[pixel_y, pixel_x]
        
        if dist < self.avoid_dis:
            # 获取梯度方向
            grad = np.array([self.grad_map[0, pixel_y, pixel_x], self.grad_map[1, pixel_y, pixel_x]])
            
            # 排斥力计算（距离越近力越大）
            # repulsive_force = self.repulsive_gain * (1/dist - 1/self.safety_radius) * grad
            repulsive_force = self.gain_a_static * np.exp( (0.5*self.safe_dis - dist)/ self.gain_b_static ) * grad
        else:
            repulsive_force = np.zeros(2)
        
        if len(nbrs_idx) != 0:
            relative_positions = np.asarray(nbrs_relpos, dtype=np.float32)
            safe_distances = np.maximum(nbrs_dis, 1e-6).reshape(-1, 1)
            away_now = -relative_positions / safe_distances
            interact_force = self.gain_a_agent * np.exp(
                (self.safe_dis - safe_distances) / self.gain_b_agent
            ) * away_now
            interact_force = np.sum(interact_force, axis=0)
        else:
            interact_force = np.zeros(2)

        # Predict closest approach using relative velocity.  relpos is
        # neighbor-minus-ego and relvel is neighbor-minus-ego velocity, so
        # relpos + relvel*t is the future separation vector.
        if len(nbrs_idx) != 0:
            relative_positions = np.asarray(nbrs_relpos, dtype=np.float32)
            relvel_sq = np.sum(nbrs_relvel * nbrs_relvel, axis=1)
            approaching = np.sum(
                relative_positions * nbrs_relvel, axis=1
            ) < 0.0
            closest_time = np.zeros(len(nbrs_idx), dtype=np.float32)
            moving = relvel_sq > 1e-6
            closest_time[moving] = np.clip(
                -np.sum(
                    relative_positions[moving] * nbrs_relvel[moving], axis=1
                )
                / relvel_sq[moving],
                0.0,
                self.prediction_horizon,
            )
            closest = relative_positions + nbrs_relvel * closest_time[:, None]
            closest_distance = np.linalg.norm(closest, axis=1)
            distance_risk = np.clip(
                (self.safe_dis - closest_distance) / max(self.safe_dis, 1e-6),
                0.0,
                1.0,
            )
            time_risk = np.clip(
                (self.ttc_threshold - closest_time)
                / max(self.ttc_threshold, 1e-6),
                0.0,
                1.0,
            )
            risk = distance_risk * time_risk * approaching.astype(np.float32)
            away = -closest / np.maximum(closest_distance[:, None], 1e-6)
            preferred_norm = max(float(np.linalg.norm(direction_to_goal)), 1e-6)
            preferred_direction = direction_to_goal / preferred_norm
            pass_right = np.asarray(
                [preferred_direction[1], -preferred_direction[0]],
                dtype=np.float32,
            )
            away[closest_distance < 1e-4] = pass_right
            ttc_force = self.gain_ttc * np.sum(risk[:, None] * away, axis=0)
        else:
            ttc_force = np.zeros(2)

        # ===== 合力合成 =====
        # Sum of push & pull forces
        d_vel = self.gain_k * (attractive_force - vel)
        interaction_vel = repulsive_force + interact_force + ttc_force
        total_d_vel = (d_vel + interaction_vel) * self.dt
        new_vel = vel + total_d_vel
        # Clamp speed to max_speed to prevent explosion when agents are very
        # close (exponential repulsion can produce huge forces).
        speed = np.linalg.norm(new_vel)
        if speed > self.max_speed:
            new_vel = new_vel * (self.max_speed / speed)

        return new_vel, [interact_force, repulsive_force, d_vel, ttc_force]
    
    def get_action(self, ego_state, env_obs):

        return self.compute_forces(ego_state, env_obs)


def load_image(img_path,scale=1/3):
    """加载并预处理地图"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    _, binary_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    binary_img = cv2.resize(binary_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return binary_img

# 可视化函数
def visualize_simulation(distance_field, path, start, goal):
    plt.imshow(distance_field, cmap='gray_r')
    plt.plot(start[1], start[0], 'gx')
    plt.plot(goal[1], goal[0], 'rx')
   
    # 绘制路径
    path = np.array(path)
    plt.plot(path[:,1], path[:,0])
    
    plt.show()

def visualize_gradient(distance_field, grad_map, stride=10, scale=30):
    """
    可视化梯度场
    :param distance_field: 距离场矩阵
    :param grad_x: x方向梯度分量
    :param grad_y: y方向梯度分量 
    :param stride: 箭头间隔像素数（控制密度）
    :param scale: 箭头缩放系数（控制长度）
    """
    
    # 假设 map_data 的形状为 (rows, cols)
    rows, cols = map_data.shape

    # 生成网格时交换维度顺序
    x_grid, y_grid = np.mgrid[0:rows:1, 0:cols:1]  # x对应行，y对应列
    
    # 下采样
    mask = (x_grid % stride == 0) & (y_grid % stride == 0)
    x_sub = x_grid[mask]
    y_sub = y_grid[mask]
    
    # 提取梯度分量（已按坐标系方向调整）
    gx_sub = grad_map[x_sub, y_sub]
    gy_sub = grad_map[x_sub, y_sub]
    
    # 绘制时翻转y轴方向（保持原点在左上角）
    plt.imshow(distance_field, cmap='gray_r')
    # plt.quiver(y_sub, x_sub, gy_sub, gx_sub,  # 交换x/y显示位置
    #            angles='xy', scale_units='xy', scale=scale,
    #            color='red', width=0.002)
    

if __name__ == "__main__":
    # 初始化环境
    image_path = Path(__file__).resolve().parents[1] / "maps" / "World0.png"
    map_data = load_image(image_path)
    sfm = Social_Force(map_data)
   
    
    # 设置起点和终点
    start = np.array([160.0, 170.0])    # 起始坐标（x,y）
    goal = np.array([680, 600])/3   # 目标坐标（x,y）

    plt.figure()
    
    
    # 运动模拟参数
    dt = 0.1           # 时间步长
    max_steps = 1000    # 最大迭代次数
    position = start.copy()
    path = [position.copy()]
    
    # 开始仿真
    for _ in range(max_steps):
        # 计算作用力
        force,af,rf = sfm.compute_forces(position, goal)
        
        # 更新位置
        position += force * dt
        
        # 记录路径
        path.append(position.copy())
        
        # 可视化结果
        plt.cla()
        plt.arrow(position[1], position[0], af[1], af[0], color='red')
        plt.arrow(position[1], position[0], rf[1], rf[0], color='blue')
        plt.arrow(position[1], position[0], force[1], force[0], color='green')


        # 终止条件检查
        if np.linalg.norm(position - goal) < 1.0:
            print("Goal reached!")
            break

    visualize_simulation(sfm.distance_field, path, start, goal)
