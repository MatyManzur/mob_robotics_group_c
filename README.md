# Mobile Robotics SS25 - Group C

## Students
- Josip Rojtinić​
- Matías Manzur

## How to run
Copy this repository contents in `catkin_ws/src/fhtw/mob_robotics_group_c` folder of your ROS Docker container. You may need to execute `roscd && chmod u+x setup.sh && ./setup.sh`.
Finally, you can execute the simulation with `roslaunch soar_maze_escape launchSimulation.launch`.

RViz will open, where you can see the robot in real-time. The global path is shown in green, the next goal relative to the robot is shown as a red arrow, and the local plan path is shown in orange.

## Project description
In `src/` folder, the code for the simulation can be found through the following files:
- `global_planner.py`: contains the code to read the map, generate the graph with the map nodes, caluclate the global path using BFS and publish it to a rostopic.
- `local_planner.py`: contains the code that gets the global path and robot position, and continuously publishes the velocity commands for the Robot to use in the simulation. It calculates the best pair of linear and angular velocities for each frame using PT2.
- `utils.py`: contains shared code between the global and local planner, such as common data structures or functions that were necessary in both stages.
- `parameters.yaml`: allows easy customization of parameters used in the simulation. Parameters are explained in comments.

