import rospy
import numpy as np
import numpy.typing as npt
import copy
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
import tf2_ros
from scipy.spatial.transform import Rotation as R

GLOBAL_PATH_TOPIC = '/move_base/global_path'
GOAL_REACHED_THRESHOLD = 0.08
COST_WEIGHTS = {
    'x': 1.0,
    'y': 1.0,
    'theta': 0.001,
    'v': 0.1,
    'w': 0.001
}
RATE_HZ = 5
HORIZON_IN_SECONDS = 2.0
HORIZON = int(RATE_HZ * HORIZON_IN_SECONDS)
CONTROL_PARAMS = {
    'min_v': -0.2,
    'max_v': 0.2,
    'min_w': -10,
    'max_w': 10,
    'dv': 0.1,
    'dw': 0.5,
    'max_dv': 0.3,
    'max_dw': 15
}

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

def pose2tf_mat(pose):
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, x],
        [s, c, y],
        [0, 0, 1]
    ])

def tf_mat2pose(tf_mat):
    x = tf_mat[0, 2]
    y = tf_mat[1, 2]
    theta = np.arctan2(tf_mat[1, 0], tf_mat[0, 0])
    return np.array([x, y, theta])

def localiseRobot(tf_buffer):
    try:
        trans = tf_buffer.lookup_transform('map', 'base_footprint', rospy.Time(0), rospy.Duration(1.0))
        x = trans.transform.translation.x
        y = trans.transform.translation.y
        theta = R.from_quat([
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w
        ]).as_euler('xyz')[2]
        rospy.loginfo("Robot pose (TF): x={:.2f}, y={:.2f}, theta={:.2f}".format(x, y, theta))
        return np.array([x, y, theta])
    except Exception as e:
        rospy.logwarn("Failed to get TF: {}".format(e))
        return None
    


def generate_controls(previous_control: np.ndarray, min_v, max_v, min_w, max_w, dv, dw, max_dv, max_dw) -> np.ndarray:
    v = previous_control[0]
    w = previous_control[1]

    possible_v = np.arange(max(v - max_dv, min_v), min(v + max_dv, max_v), dv)
    possible_w = np.arange(max(w - max_dw, min_w), min(w + max_dw, max_w), dw)
    print(f"Possible v range: ({possible_v[0]}, {possible_v[-1]})")
    print(f"Possible w range: ({possible_w[0]}, {possible_w[-1]})")
    controls = []
    for v in possible_v:
        for w in possible_w:
            controls.append(np.array([v, w]))
    print(f"Generated {len(controls)} control signals")
    return np.array(controls)


class LocalPlanner:
    def __init__(self):
        rospy.init_node('local_planner')
        self.path_sub = rospy.Subscriber(GLOBAL_PATH_TOPIC, Path, self.path_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.goal_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)
        self.traj_publisher = rospy.Publisher(name='/trajectory', data_class=Path, queue_size=400)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.global_path = []
        self.last_control = np.array([0.0, 0.0])

    def publishCMDVel(self, control: np.ndarray):
        """Publish control signal to robot
        """
        cmd = Twist()
        cmd.linear.x = control[0]
        cmd.angular.z = control[1]
        rospy.loginfo("Publishing control: v={:.6f}, w={:.6f}".format(cmd.linear.x, cmd.angular.z))
        self.cmd_vel_pub.publish(cmd)

        
    def forwardKinematics(self, control: npt.ArrayLike, lastPose: npt.ArrayLike, dt: float, dtype=np.float64) -> np.ndarray:
        """Mobile robot forward kinematics (see Thrun Probabilistic Robotics)
        """
        if not isinstance(lastPose, np.ndarray):  # Check input formatting
            lastPose = np.array(lastPose, dtype=dtype)
        assert lastPose.shape == (3,), "Wrong pose format. Pose must be provided as list or array of form [x, y, theta]"
        if not isinstance(control, np.ndarray): 
            control = np.array(control)
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

    def evaluateControls(self, controls, robotModelPT2, horizon, goal, ts, vis_horizon=70):
        costs = np.zeros_like(np.array(controls)[:,0], dtype=float)
        trajectories = [ [] for _ in controls ]
        
        # Apply range of control signals and compute outcomes
        for ctrl_idx, control in enumerate(controls):
        
            # Copy currently predicted robot state
            forwardSimPT2 = copy.deepcopy(robotModelPT2)
            forwardpose = [0,0,0]
        
            # Simulate until horizon
            for step in range(vis_horizon):
                control_sim = copy.deepcopy(control)
                v_t, w_t = control
                v_t_dynamic = forwardSimPT2.update(v_t)
                control_dym = [v_t_dynamic, w_t]
                forwardpose = self.forwardKinematics(control_dym, forwardpose, ts)
                if step < horizon:
                    costs[ctrl_idx] += self.costFn(forwardpose, goal, control_sim)
                # Track trajectory for visualisation
                trajectories[ctrl_idx].append(forwardpose)

        return costs, trajectories

    def costFn(self, pose: npt.ArrayLike, goalpose: npt.ArrayLike, control: npt.ArrayLike) -> float:
        diff = pose - goalpose
        diff[2] += 2*np.pi*np.floor((np.pi - diff[2]) / (2*np.pi))
        e = np.abs(diff)
        Q = np.diag([COST_WEIGHTS['x'], COST_WEIGHTS['y'], COST_WEIGHTS['theta']])
        R = np.diag([COST_WEIGHTS['v'], COST_WEIGHTS['w']])
        u = np.abs(control)
        cost = (e.T @ (Q @ e)) + (u.T @ (R @ u))
        return cost

    def publishGoal(self, goalpose: np.ndarray):
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.x = goalpose[0]
        goal_msg.pose.position.y = goalpose[1]
        goal_msg.pose.orientation.w = np.cos(goalpose[2] / 2)
        goal_msg.pose.orientation.z = np.sin(goalpose[2] / 2)
        self.goal_pub.publish(goal_msg)
    
    def publishTrajectory(self, trajectory: npt.ArrayLike):
        path = Path()
        path.header.frame_id = 'base_footprint'
        path.header.stamp = rospy.Time.now()
        for pose in trajectory:
            pose_stamped = PoseStamped()
            pose_stamped.header = path.header
            pose_stamped.pose.position.x = pose[0]
            pose_stamped.pose.position.y = pose[1]
            pose_stamped.pose.orientation.w = np.cos(pose[2] / 2)
            pose_stamped.pose.orientation.z = np.sin(pose[2] / 2)
            path.poses.append(pose_stamped)
        self.traj_publisher.publish(path)

    def path_callback(self, msg: Path): # When the path is received, we start the control loop
        self.global_path = []
        for pose in msg.poses:
            rotation = R.from_quat([
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w
            ]).as_euler('xyz')[2]
            self.global_path.append([pose.pose.position.x, pose.pose.position.y, rotation])
        rospy.loginfo("Received path with {} points".format(len(self.global_path)))
        if self.global_path:
            self.control_loop()

    def control_loop(self):
        rate = rospy.Rate(RATE_HZ)
        current_goal_id = 0

        while not rospy.is_shutdown() and current_goal_id < len(self.global_path):
            # We localize the robot
            robot_pose = localiseRobot(self.tf_buffer)
            
            if robot_pose is None:
                rospy.logwarn("Cannot localize robot, skipping control loop")
                rate.sleep()
                continue

            goal_pose = self.global_path[current_goal_id]
            rospy.loginfo("Current goal {}: x={:.2f}, y={:.2f}".format(current_goal_id, goal_pose[0], goal_pose[1]))

            robot_tf = pose2tf_mat(robot_pose)
            goal_tf = pose2tf_mat(goal_pose)
            relative_tf = np.linalg.inv(robot_tf) @ goal_tf
            goal_from_robot = tf_mat2pose(relative_tf)
            rospy.loginfo("Relative goal: x={:.2f}, y={:.2f}, theta={:.2f}".format(goal_from_robot[0], goal_from_robot[1], goal_from_robot[2]))

            distance_to_goal = np.linalg.norm(goal_from_robot[:2])

            if distance_to_goal < GOAL_REACHED_THRESHOLD:
                rospy.loginfo("Goal {} reached!".format(current_goal_id))
                current_goal_id += 1
                continue

            rospy.loginfo("Distance to goal: %.2f" % (distance_to_goal,))
            
            controls = generate_controls(
                self.last_control, 
                min_v=CONTROL_PARAMS['min_v'], 
                max_v=CONTROL_PARAMS['max_v'], 
                min_w=CONTROL_PARAMS['min_w'], 
                max_w=CONTROL_PARAMS['max_w'], 
                dv=CONTROL_PARAMS['dv'], 
                dw=CONTROL_PARAMS['dw'], 
                max_dv=CONTROL_PARAMS['max_dv'], 
                max_dw=CONTROL_PARAMS['max_dw'])
            
            robotModelPT2 = PT2Block(ts=(1/RATE_HZ), T=0.05, D=0.8)
            costs, trajectories = self.evaluateControls(controls, robotModelPT2, horizon=HORIZON, goal=goal_from_robot, ts=(1/RATE_HZ), vis_horizon=HORIZON)

            idx = np.argmin(costs)

            #print(f"Index with lowest cost: {idx}")
            #print(f"Resulting cost: {costs[idx]}")
            #print(f"Resulting control: {controls[idx]}")
            new_control = controls[idx]
            if np.all(new_control == 0): # if the control is zero, we select the second lowest cost
                costs[idx] = np.inf  # Remove the lowest cost by setting it to infinity
                idx = np.argmin(costs)  # Find the new minimum
                new_control = controls[idx]

            self.last_control = new_control

            self.publishCMDVel(controls[idx])
            self.publishGoal(goal_pose)
            self.publishTrajectory(trajectories[idx])

            if current_goal_id + 1 < len(self.global_path) and np.linalg.norm(goal_from_robot[:2]) < GOAL_REACHED_THRESHOLD:
                current_goal_id += 1

            rate.sleep()

        rospy.loginfo("All goals reached! Stopping robot.")
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

if __name__ == '__main__':
    try:
        LocalPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass