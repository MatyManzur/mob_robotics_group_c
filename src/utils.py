from geometry_msgs.msg import Point, PoseStamped, Pose, Quaternion
from nav_msgs.msg import Path
from typing import Optional, List
import math, tf, rospy
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R

class MapPoint():
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __hash__(self):
        return hash((self.x, self.y))
    def __str__(self):
        return f"({self.x}, {self.y})"
    
class Map():
    def __init__(self, resolution, origin):
        self.resolution = resolution
        self.origin = origin

    def map_to_world_position(self, point: MapPoint) -> Point:
        """ Converts map coordinates to world coordinates """
        x = point.x
        y = point.y
        _x = (x + 0.5) * self.resolution + self.origin[0]
        _y = (y + 0.5) * self.resolution + self.origin[1]
        return Point(_x, _y, 0.0)

    def world_to_map_position(self, x: float, y: float) -> MapPoint:
        """ Converts world coordinates to map coordinates """
        x = (x - self.origin[0]) / self.resolution - 0.5
        y = (y - self.origin[1]) / self.resolution - 0.5
        return MapPoint(int(x), int(y))

def world_point_to_pose(point: Point, next_point: Optional[Point]) -> Pose:
    """ Converts a world point to a Pose """
    pose = Pose()
    pose.position = point
    if next_point is not None:
        dx = next_point.x - point.x
        dy = next_point.y - point.y
        yaw = math.atan2(dy, dx)
        pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, yaw))
    else:
        pose.orientation = Quaternion(0, 0, 0, 1)
    return pose

def world_points_to_path(points: List[Point], time: rospy.Time) -> Path:
    """ Converts a list of world points to a Path """
    path = Path()
    path.header.frame_id = "map"
    path.header.stamp = time
    for i in range(len(points)):
        pose = PoseStamped()
        pose.header = path.header
        pose.pose = world_point_to_pose(points[i], points[i + 1] if i < len(points) - 1 else None)
        path.poses.append(pose)
    return path


def localiseRobot(tfBuffer: tf2_ros.Buffer) -> np.ndarray:
    listener = tf2_ros.TransformListener(tfBuffer)
    """Localises the robot towards the 'map' coordinate frame. Returns pose in format (x,y,theta)"""
    while True:
        try:
            trans = tfBuffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("Robot localisation took longer than 1 sec")
            continue

    theta = R.from_quat([
        trans.transform.rotation.x,
        trans.transform.rotation.y,
        trans.transform.rotation.z,
        trans.transform.rotation.w]).as_euler("xyz")[2]
    
    return np.array([
        trans.transform.translation.x,
        trans.transform.translation.y,
        theta])