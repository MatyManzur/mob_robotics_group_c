import rospy
import tf2_ros
import numpy as np
import numpy.typing as npt
import copy

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path

rospy.init_node("local_planner") 

tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)

def localiseRobot():
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

def pose2tf_mat(pose: np.ndarray) -> np.ndarray:
    x, y, theta = pose
    tf_mat = np.eye(3)
    tf_mat[0, 2] = x
    tf_mat[1, 2] = y
    tf_mat[0, 0] = np.cos(theta)
    tf_mat[0, 1] = -np.sin(theta)
    tf_mat[1, 0] = np.sin(theta)
    tf_mat[1, 1] = np.cos(theta)
    return tf_mat

def tf_mat2pose(tf_mat: np.ndarray) -> np.ndarray:
    x = tf_mat[0, 2]
    y = tf_mat[1, 2]
    theta = np.arctan2(tf_mat[1, 0], tf_mat[0, 0])
    return np.array([x, y, theta])

robot = localiseRobot()
goal = np.array([3, 1.5, 0.5*np.pi])

robot_mat = pose2tf_mat(robot)
goal_mat = pose2tf_mat(goal)
robot_mat_inv = np.linalg.inv(robot_mat)

goal_from_robot = tf_mat2pose(robot_mat_inv @ goal_mat)
print("Robot pose in map frame: ", robot)
print(goal_from_robot)

def generate_controls(previous_control: np.ndarray, min_v, max_v, min_w, max_w, dv, dw, max_dv, max_dw) -> np.ndarray:
    v = previous_control[0]
    w = previous_control[1]

    possible_v = np.arange(max(v - max_dv, min_v), min(v + max_dv, max_v), dv)
    possible_w = np.arange(max(w - max_dw, min_w), min(w + max_dw, max_w), dw)
    controls = []
    for v in possible_v:
        for w in possible_w:
            controls.append(np.array([v, w]))
    return np.array(controls)

last_control = np.array([0, 0])
controls = generate_controls(last_control, 
                             min_v=-0.025, 
                             max_v=0.01, 
                             min_w=-1.5, 
                             max_w=1.5, 
                             dv=0.0115, 
                             dw=0.025, 
                             max_dv=0.0325, 
                             max_dw=1.4)

def forwardKinematics(control: npt.ArrayLike, lastPose: npt.ArrayLike, dt: float, dtype=np.float64) -> np.ndarray:
    """Mobile robot forward kinematics (see Thrun Probabilistic Robotics)
    """
    if not isinstance(lastPose, np.ndarray):  # Check input formatting
        lastPose = np.array(lastPose, dtype=dtype)
    assert lastPose.shape == (3,), "Wrong pose format. Pose must be provided as list or array of form [x, y, theta]"
    if not isinstance(control, np.ndarray): control = np.array(control)
    assert control.shape == (2,), "Wrong control format. Control must be provided as list or array of form [vt, wt]"
    vt, wt = control
    # Set omega to smallest possible nonzero value in case it is zero to avoid division by zero
    if wt == 0: wt = np.finfo(dtype).tiny
    vtwt = vt/wt
    _, _, theta = lastPose
    return lastPose + np.array([
        -vtwt*np.sin(theta) + vtwt*np.sin(theta + (wt*dt)),
        vtwt*np.cos(theta) - vtwt*np.cos(theta + (wt*dt)),
        wt*dt
    ], dtype=dtype)

class PT2Block:
    """Discrete PT2 Block approximated using the Tustin approximation (rough robot dynamics model)
    """
    def __init__(self, T=0, D=0, kp=1, ts=0, bufferLength=3) -> None:
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6 = 0, 0, 0, 0, 0, 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0:  self.setConstants(T, D, kp, ts)
    #
    def setConstants(self, T, D, kp, ts) -> None:
        self.k1 = 4*T**2 + 4*D*T*ts + ts**2
        self.k2 = 2*ts**2 - 8*T**2
        self.k3 = 4*T**2 - 4*D*T*ts + ts**2
        self.k4 = kp*ts**2
        self.k5 = 2*kp*ts**2
        self.k6 = kp*ts**2
    #
    def update(self, e) -> float:    
        self.e = [e]+self.e[:len(self.e)-1] # Update buffered input and output signals
        self.y = [0]+self.y[:len(self.y)-1]
        e, y = self.e, self.y # Shorten variable names for better readability
        # Calculate output signal and return output
        y[0] = ( e[0]*self.k4 + e[1]*self.k5 + e[2]*self.k6 - y[1]*self.k2 - y[2]*self.k3 )/self.k1
        return y[0]

def costFn(pose: npt.ArrayLike, goalpose: npt.ArrayLike, control: npt.ArrayLike) -> float:
    diff = pose - goalpose
    diff[2] = max(-np.pi, min(np.pi, diff[2]))
    e = np.abs(diff)
    Q = np.diag([1, 1, 0.5])
    R = np.diag([0.1, 0.1])
    u = np.abs(control)
    cost = e.T @ (Q @ e) + u.T @ (R @ u)
    return cost

def evaluateControls(controls, robotModelPT2, horizon):
    costs = np.zeros_like(np.array(controls)[:,0], dtype=float)
    trajectories = [ [] for _ in controls ]
    
    # Apply range of control signals and compute outcomes
    for ctrl_idx, control in enumerate(controls):
    
        # Copy currently predicted robot state
        forwardSimPT2 = copy.deepcopy(robotModelPT2)
        forwardpose = [0,0,0]
    
        # Simulate until horizon
        for step in range(horizon):
            control_sim = copy.deepcopy(control)
            v_t, w_t = control
            v_t_dynamic = forwardSimPT2.update(v_t)
            control_dym = [v_t_dynamic, w_t]
            forwardpose = forwardKinematics(control_dym, forwardpose, ts)
            costs[ctrl_idx] += costFn(forwardpose, goal, control_sim)
            # Track trajectory for visualisation
            trajectories[ctrl_idx].append(forwardpose)

    return costs, trajectories

ts = 1/2 # Sampling time [sec] -> 2Hz
horizon = 10 # Number of time steps to simulate. 10*0.5 sec = 5 seconds lookahead into the future
robotModelPT2 = PT2Block(ts=ts, T=0.05, D=0.8)
costs, trajectories = evaluateControls(controls, robotModelPT2, 70) # -> We sample 70 control updates here to make the visualisation later 

idx = np.argmin(costs)

print(f"Index with lowest cost: {idx}")
print(f"Resulting cost: {costs[idx]}")
print(f"Resulting control: {controls[idx]}")

cmd_publisher = rospy.Publisher(name='/cmd_vel', data_class=Twist, queue_size=1)
traj_publisher = rospy.Publisher(name='/trajectory', data_class=Path, queue_size=1)
goal_publisher = rospy.Publisher(name='/local_goal', data_class=PoseStamped, queue_size=1)

def publishCMDVel(control: np.ndarray):
    """Publish control signal to robot
    """
    cmd = Twist()
    cmd.linear.x = control[0]
    cmd.angular.z = control[1]
    cmd_publisher.publish(cmd)

