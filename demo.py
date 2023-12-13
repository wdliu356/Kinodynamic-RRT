import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, wait_if_gui, load_pybullet
import random
import pybullet as p
import time
class Node:
    def __init__(self,config_in):
        self.config = config_in
        self.parent = None
        # self.children = []



np.random.seed(1)
state_names = ('x', 'y', 'theta', 'u', 'v', 'w')
state_limits = {'x': (-4.75, 4.75), 'y': (-4.75, 4.75), 'theta': (-np.pi, np.pi), 'u': (-0.5, 0.5),
                'v': (-0.5, 0.5), 'w': (-0.3, 0.3)}


def distance(config1, config2):
    weight = np.array([0.35,0.35,0.15,0.05,0.05,0.05])
    ##config is a tuple
    array1 = np.array(config1)
    array2 = np.array(config2)
    return np.linalg.norm((array1 - array2)*weight)

def get_closest_node(rand_config, node_list):
    min_dist = float('inf')
    closest_node = None
    for node in node_list:
        dist = distance(node.config, rand_config)
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    return closest_node

def dynamic_model(config, input, dt):
    #config is the state configuration
    #input is the control input including f_u, f_v, f_w
    #dt is the time step
    #return the next state configuration
    
    x, y, theta, u, v, w = config
    f_u, f_v, f_w = input
    m = 1
    I = 1
    x_next = x + u *dt * np.cos(theta) - v * dt * np.sin(theta)
    y_next = y + u *dt * np.sin(theta) + v * dt * np.cos(theta)
    theta_next = theta + w * dt
    if theta_next > np.pi:
        theta_next -= 2 * np.pi
    elif theta_next < -np.pi:
        theta_next += 2 * np.pi
    u_next = u + f_u / m * dt
    u_next = min(u_next, state_limits['u'][1])
    u_next = max(u_next, state_limits['u'][0])
    v_next = v + f_v / m * dt
    v_next = min(v_next, state_limits['v'][1])
    v_next = max(v_next, state_limits['v'][0])
    w_next = w + f_w / I * dt
    w_next = min(w_next, state_limits['w'][1])
    w_next = max(w_next, state_limits['w'][0])

    return (x_next, y_next, theta_next, u_next, v_next, w_next)

def get_new_config(current_config, rand_config,input_sample_num,collision_fn, dt):
    input_list = []
    input_list.append((1,0,0))##move forward
    input_list.append((-1,0,0))##move backward
    input_list.append((0,0,1))##turn left
    input_list.append((0,0,-1))##turn right
    input_list.append((-1,-1,0))##move backward and right
    input_list.append((1,0,1))##move forward and turn left
    input_list.append((1,0,-1))##move forward and turn right
    input_list.append((-1,0,1))##move backward and turn left
    input_list.append((-1,0,-1))##move backward and turn right
    ## 0 - 8 are the 9 basic motions, pretending the robot is a car
    input_list.append((0,1,0))##move left
    input_list.append((0,-1,0))##move right
    input_list.append((1,1,0))##move forward and left
    input_list.append((1,-1,0))##move forward and right
    input_list.append((-1,1,0))##move backward and left
    input_list.append((0,1,1))##move left and turn left
    input_list.append((0,1,-1))##move left and turn right
    input_list.append((0,-1,1))##move right and turn left
    input_list.append((0,-1,-1))##move right and turn right
    input_list.append((1,1,1))##move forward and left and turn left
    input_list.append((1,1,-1))##move forward and left and turn right
    input_list.append((1,-1,1))##move forward and right and turn left
    input_list.append((1,-1,-1))##move forward and right and turn right
    input_list.append((-1,1,1))##move backward and left and turn left
    input_list.append((-1,1,-1))##move backward and left and turn right
    input_list.append((-1,-1,1))##move backward and right and turn left
    input_list.append((-1,-1,-1))##move backward and right and turn right
    # input_list*=100
    ## 9 - 24 are the 16 basic motions, pretending the robot is a car
    dis = []
    new_config_list = []
    for i in range(input_sample_num):
        new_config = dynamic_model(current_config, input_list[i], dt)
        ## check whether the new config is within the range before adding to the list
        if not limit_check(new_config) or collision_fn((new_config[0], new_config[1], new_config[2])):
            continue
        dis.append(distance(new_config, rand_config))
        new_config_list.append(new_config)
    if len(dis) == 0:
        return None
    min_dis = min(dis)
    min_index = dis.index(min_dis)
    return new_config_list[min_index]

def limit_check(config):
    for i in range(6):
        if (config[i] < state_limits[state_names[i]][0] or config[i] > state_limits[state_names[i]][1]) and i != 2:
            return False
    return True


def expand_node(node_list, closest_node, rand_config, collision_fn, input_sample_num, dt):
    new_node_list = node_list
    current_node = closest_node
    num = 0
    while distance(current_node.config, rand_config) > 0.1 and num < 100:
        new_config = get_new_config(current_node.config, rand_config,input_sample_num,collision_fn, dt)
        if new_config == None:
            break
        new_node = Node(new_config)
        new_node.parent = current_node
        new_node_list.append(new_node)
        current_node = new_node
        num += 1
        

    return new_node_list

def rrt_connect(start_config, goal_config, collision_fn, input_sample_num):
    goal_bias = 0.1
    dt = 0.1
    max_iter = 100000
    start_node = Node(start_config)
    goal_node = Node(goal_config)
    node_list = [start_node]
    path = []
    find_path = False
    for i in range(max_iter):
        if random.random() < goal_bias:
            rand_config = goal_config
        else:
            rand_config = [random.uniform(state_limits[name][0], state_limits[name][1]) for name in state_names]
            while not limit_check(rand_config) or collision_fn((rand_config[0], rand_config[1], rand_config[2])):
                rand_config = [random.uniform(state_limits[name][0], state_limits[name][1]) for name in state_names]
        closest_node = get_closest_node(rand_config, node_list)
        node_list = expand_node(node_list, closest_node, rand_config, collision_fn, input_sample_num, dt)
        if distance(node_list[-1].config, goal_config) < 0.1 or ((node_list[-1].config[0]-goal_config[0])**2 < 0.03 and (node_list[-1].config[1]-goal_config[1])**2 < 0.03 and (node_list[-1].config[2]-goal_config[2])**2 < 0.05) :
            goal_node.parent = node_list[-1]
            node_list.append(goal_node)
            find_path = True
            break
    if find_path:
        path = get_path(node_list, goal_node)
    return node_list, path

def get_path(node_list, goal_node):
    path = []
    current_node = goal_node
    while current_node.parent != None:
        path.append((current_node.config[0], current_node.config[1], current_node.config[2]))
        current_node = current_node.parent
    path.append((current_node.config[0], current_node.config[1], current_node.config[2]))
    path.reverse()
    return path

def path_quality(explored, path):
    totalnodes = len(explored)
    nodenum = len(path)
    quality_1 = 0
    quality_2 = 0
    for i in range(nodenum-1):
        quality_1 += np.sqrt((path[i][0]-path[i+1][0])**2+(path[i][1]-path[i+1][1])**2)
        quality_2 += np.sqrt((path[i][0]-path[i+1][0])**2+(path[i][1]-path[i+1][1])**2+(path[i][2]-path[i+1][2])**2)
    return totalnodes, nodenum, quality_1, quality_2
def draw_sphere_markers(configs, color,z, marker_size=3.):
    """
    Args:
        configs (list): (N, dim_config) in task space [X, Y, ...]
        color (list): (3, ) in RGB
        marker_size (float): marker size
    """
    num_point = len(configs)
    if num_point == 0:
        return

    config_array = np.array(configs) # (N, dim_config) config should start with x and y
    # if num_point > MAX_POINTS:
    #     print(f"Too many points ({num_point}) to draw, will downsample to {MAX_POINTS}.")
    #     num_point = MAX_POINTS
    #     config_array = RNG.choice(config_array, num_point, replace=False)

    point_positions = np.zeros((num_point, 3))
    point_positions[:, :2] = config_array[:, :2] # x, y
    point_positions[:, -1] = z # z
    p.addUserDebugPoints(pointPositions=point_positions, 
                         pointColorsRGB=[color[:3] for _ in range(num_point)], 
                         pointSize=marker_size, 
                         lifeTime=0)

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=False)
    # load robot and obstacle resources
    _, obstacles = load_env('environment.json')
    robot = load_pybullet("myrobot.urdf")
    # define active DoFs
    base_joints = [0, 1, 2]
    collision_fn = get_collision_fn_PR2(robot, base_joints, list(obstacles.values()))
    print()
    print()
    print("Building RRT...")
    print("We will show the search trees(blue) and executed path(green) in this program.")
    print("This program is expected to run for 4~10 minutes (incluing drawing paths)...")
    print()
    start_config = (-4.5, 4.5, 0, 0, 0, 0)
    goal_config = (4.5, -4.5, -np.pi/2, 0, 0, 0)
    draw_sphere_marker((goal_config[0], goal_config[1], 0.1), 0.15, (1, 0, 0, 1))
    start_time = time.time()
    explored, path = rrt_connect(start_config, goal_config, collision_fn, 9)
    totalnodes, nodenum, quality_eu, quality_plus = path_quality(explored, path)
    print("Finish building the RRT!!")
    print("computation time:", time.time() - start_time)
    print("number of explored nodes:", totalnodes)
    print("number of path nodes:", nodenum)
    print("path quality(euclidean):", quality_eu)
    print("path quality(consider theta):", quality_plus)
    explored_config = []
    for node in explored:
        explored_config.append(node.config)
    disconnect()
    connect(use_gui=True)
    _, obstacles = load_env('environment.json')
    robot = load_pybullet("myrobot.urdf")
    start_config = (-4.5, 4.5, 0)
    collision_fn(start_config)
    draw_sphere_marker((goal_config[0], goal_config[1], 0.1), 0.15, (1, 0, 0, 1))
    print("Start drawing the explored path (Blue)")
    # for path_i in explored:
    #     draw_sphere_marker((path_i.config[0], path_i.config[1], 0.06), 0.05, (0, 0, 1, 1))
    draw_sphere_markers(explored_config, (0, 0, 1),0.1, marker_size=3)
    print("Start drawing the path (green)")
    draw_sphere_markers(path, (0, 1, 0), 0.3, marker_size=3)
    # for path_i in path:
    #     draw_sphere_marker((path_i[0], path_i[1], 0.08), 0.05, (0, 1, 0, 1))

    ######################
    print("Start executing the path")
    # Execute planned path
    execute_trajectory(robot, base_joints, path, sleep=0.1)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()


if __name__ == '__main__':
    main()
