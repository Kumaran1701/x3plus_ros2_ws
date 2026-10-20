# X3Plus ROS 2 Workspace

Workspace for autonomous manipulation with the Yahboom X3Plus mobile
manipulator. The long-term project is organized as three stages: navigate the
mobile base to a task location, perceive the local scene and plan a
collision-aware arm approach, then perceive and execute the manipulation task.
The current implementation focuses on the second stage.

## Project stages

1. **Mobile motion and localization to the target — future work.** The robot
   needs to navigate to the task location and localize relative to it.
   Autonomous navigation and localization are not implemented yet; Nav2 with
   the robot's lidar is the likely direction. The existing base strafe is a
   controlled motion used for scene capture, not autonomous navigation.
2. **Collision-aware motion planning to the task — implemented and run on hardware.** RGB-D     perception provides scene geometry for collision checking.
   Drake's IRIS/GCS workflow plans the arm to a pre-task pose, and ROS 2 nodes
   stream the trajectory to the robot.
3. **Manipulation-task execution — next stage.** Future work includes
   perceiving the target hole and executing insertion, with an enlarged hole
   tolerance to make the task more robust.

The current planner ends at the pre-task pose: target-hole perception and
insertion are not yet implemented.

## Results

Project results and demonstrations are being prepared and will be added soon.
Planned reporting includes representative obstacle reconstructions, planning
success and timing across test scenes, and results from hardware trajectory
execution. Results will include their test conditions and measurement method;
no performance or accuracy claims are made here yet.

## Package documentation

- [x3plus_bringup](x3plus_bringup/README.md)
- [x3plus_description](x3plus_description/README.md)
- [x3plus_drake_planner](x3plus_drake_planner/README.md)
- [x3plus_motion_planner](x3plus_motion_planner/README.md)
- [x3plus_perception_utils](x3plus_perception_utils/README.md)