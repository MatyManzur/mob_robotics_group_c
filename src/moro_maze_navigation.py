import rospy

from geometry_msgs.msg import Point, PoseStamped, Pose, Quaternion
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid, Path
from tf.transformations import quaternion_from_euler

import numpy as np
from utils import Map, MapPoint, world_points_to_path 

from typing import Dict, List, Tuple
from utils import Map, localiseRobot
import tf2_ros
from scipy.ndimage import binary_dilation

GLOBAL_PATH_TOPIC = '/move_base/global_path'
MIN_WALL_DISTANCE = 1

# Helper method for retrieving the map
def getMap() -> OccupancyGrid:
    """ Loads map from map service """
    # Create service proxy
    get_map = rospy.ServiceProxy('static_map', GetMap)
    # Call service
    recMap = get_map()
    recMap = recMap.map
    # Return
    return recMap

# Initiate ROS node
rospy.init_node('moro_maze_navigation')
recMap = getMap()
grid = np.split(np.array(recMap.data), recMap.info.height)
# transpose the grid
grid = np.array(grid).T
resolution = recMap.info.resolution
origin = recMap.info.origin.position
origin = np.array([origin.x, origin.y])

def get_close_to_wall_grid(grid: np.ndarray, min_distance: int) -> np.ndarray:
    binary_grid = (grid == 100)
    structuring_element = np.ones((2 * min_distance + 1, 2 * min_distance + 1), dtype=bool)
    dilated_grid = binary_dilation(binary_grid, structure=structuring_element)
    return dilated_grid

close_to_wall_grid = get_close_to_wall_grid(grid, MIN_WALL_DISTANCE)

# Printing map
for i in range(len(grid)):
    for j in range(len(grid[i])):
        if grid[i][j] == 100:
            print("X", end="")
        else:
            print(" ", end="")
    print("")

map: Map = Map(resolution, origin)

# Generate graph
graph: Dict[MapPoint, List[Tuple[MapPoint, float]]] = {}

for i in range(len(grid)):
    for j in range(len(grid[i])):
        if grid[i][j] == 100:
            continue
        point = MapPoint(i, j)
        graph[point] = []
        if i == 0 or j == 0 or i == len(grid) - 1 or j == len(grid[i]) - 1:
            continue
        for x in range(-1, 2):
            for y in range(-1, 2):
                if (x == 0 and y == 0):
                    continue
                if i + x < 0 or i + x >= len(grid) or j + y < 0 or j + y >= len(grid[i]):
                    continue
                if close_to_wall_grid[i + x][j + y]:
                    continue
                graph[point].append((MapPoint(i + x, j + y), np.sqrt(x ** 2 + y ** 2)))

tfBuffer = tf2_ros.Buffer()

# Get robot position
robot_position = localiseRobot(tfBuffer)

robot = map.world_to_map_position(robot_position[0], robot_position[1])

goal_position = (0, 0) # TODO: get goal position

goal = map.world_to_map_position(goal_position[0], goal_position[1])

# Get path to goal using DFS
def dfs(graph: Dict[MapPoint, List[Tuple[MapPoint, float]]], start: MapPoint, goal: MapPoint) -> List[MapPoint]:
    stack = [start]
    visited = set()
    parent = {}
    while len(stack) > 0:
        node = stack.pop()
        if node == goal:
            break
        if node in visited:
            continue
        visited.add(node)
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
                parent[neighbor] = node
    path = []
    while goal != start:
        path.append(goal)
        goal = parent[goal]
    path.append(start)
    path.reverse()
    return path

# Get path to goal using BFS
def bfs(graph: Dict[MapPoint, List[Tuple[MapPoint, float]]], start: MapPoint, goal: MapPoint) -> List[MapPoint]:
    queue = [start]
    visited = set()
    parent = {}
    while len(queue) > 0:
        node = queue.pop(0)
        if node == goal:
            break
        if node in visited:
            continue
        visited.add(node)
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)
                parent[neighbor] = node
    path = []
    while goal != start:
        path.append(goal)
        goal = parent[goal]
    path.append(start)
    path.reverse()
    return path

global_path: List[MapPoint] = bfs(graph, robot, goal)

# Convert path to world coordinates
global_path_world = [map.map_to_world_position(point) for point in global_path]

path_msg: Path = world_points_to_path(global_path_world, rospy.Time.now())

# Publish path
pub = rospy.Publisher(GLOBAL_PATH_TOPIC, Path, queue_size=10)
rospy.sleep(1.0)
pub.publish(path_msg)


"""
poses = []

for i in range(len(global_path)):
    world_point = map_to_world_position(global_path[i])
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = rospy.Time.now()
    pose.pose.position = world_point
    if i < len(global_path) - 1:
        next_world_point = map_to_world_position(global_path[i + 1])
        dx = next_world_point.x - world_point.x
        dy = next_world_point.y - world_point.y
        angle = np.arctan2(dy, dx)
        pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, angle))
    else:
        pose.pose.orientation = Quaternion(0, 0, 0, 1)
    poses.append(pose)

# Create path message
path_msg = Path()
path_msg.header.frame_id = 'map'
path_msg.header.stamp = rospy.Time.now()
path_msg.poses = poses

# Publish path
pub = rospy.Publisher('/global_plannnnn', Path, queue_size=10)
rospy.sleep(1.0)
pub.publish(path_msg)
"""

    