# `x3plus_description`

Robot models for the Yahboom X3Plus mobile manipulator. It gives the ROS 2 tools and the Drake planning pipeline a shared description of the robot's links, joints, sensor mounts and geometry, in two forms: the ROS 2 description used by bringup and visualization, and a reduced Drake model prepared for motion planning.

For the hardware interfaces and state estimation, see
[`x3plus_bringup`](../x3plus_bringup/README.md).

## Models

| File | Source | Used for |
| --- | --- | --- |
| `urdf/x3plus.gazebo.urdf.xacro` | Yahboom-derived | Main ROS 2 description, loaded by `x3plus_bringup` for `robot_state_publisher`: mecanum base, five-joint arm, gripper and sensor links, with STL meshes |
| `urdf/drake_x3plus_fixed.urdf` | Derived from the Yahboom URDF | Drake model of the base and arm for planning; only the five arm joints are free |
| `urdf/drake_x3plus_gripper.urdf` | Derived from the Yahboom URDF | Separate Drake gripper model, welded to `arm_link5` by the planning code |
| `meshes/**/*.obj` | Converted from Yahboom's STL meshes | Geometry referenced by the Drake URDFs |
| `urdf/yahboomcar_X3plus.urdf`, `.urdf.xacro`, `x3plus.gazebo.xacro`, `x3plus_ros2_control.xacro` | Yahboom | Original model variants and Gazebo/control fragments, kept for reference |
| `launch/rviz.launch.py`, `rviz/yahboomcar.rviz` | Yahboom | Opens RViz with the model and TF displays |

The description is kinematics and geometry only. It does not provide drivers, state estimation or planning.

## Why the Drake models differ from the ROS 2 model

The ROS 2 description is complete: every wheel and gripper joint is a movable joint. That is the right model for TF, but not for planning. The Drake models were derived from it with three changes.

**Only the five arm joints are free.** Iterative Region Infaltion by Semidefinite programming (IRIS) works in the plant's configuration space, and every movable joint adds a dimension to it. With the wheels and gripper joints left free, the plant had 22 configurable positions, most of them irrelevant to the planning problem: the base does not move while the arm is planned, and the gripper is not a planned degree of freedom. All other joints, including the wheels, are fixed, so the configuration space is only the arm's five joints and the regions IRIS builds wil be in five dimensions.

**The gripper is a separate model.** In Drake the gripper is an attachment to the arm, not part
of it, so it is kept in `drake_x3plus_gripper.urdf` and welded to `arm_link5` by the planning
code. This keeps the robot at its actual five degrees of freedom. Leaving the gripper joints
free would have made it six.

**No mimic joints, and a fixed gripper pose.** The original gripper URDF uses mimic joints, which Drake supports only in SDF models, not in URDF. They were removed rather than converting the model to SDF because the gripper is not planned degree of freedom anyway. The gripper joints are fixed in their most extended configuration, so regions grown by IRIS are valid for the gripper at its maximum extent.

**Held objects are not modeled yet.** The gripper model is empty and fixed at its largest
opening, so plans made with it do not account for anything that the gripper carries. The planned
approach is to weld a model of the known object to the gripper before finalizing the Drake MultiBodyPlant, with the gripper joints fixed at the grasp width, and to filter collisions between the object and the finger links. Regions must be regrown whenever the held object changes.

## Meshes

The ROS 2 description uses STL meshes under `package://x3plus_description/meshes/...`. The Drake
URDFs use OBJ copies of the same meshes under `package://meshes/...` because `.stl` format is not supported in Drake. The planning code maps the `meshes/` folder to the Drake package name `meshes`.

**A fresh clone is missing some of the Drake geometry.** The two `base_link_wo_wheels.obj` files
(in `meshes/X3plus/visual/` and `meshes/X3plus/collision/`) are about 153 MB each and are excluded from Git by `.gitignore`. The Drake URDFs refer to them, so the Drake model will not load until they are obtained or regenerated. The matching STL files are in the repository.

Planned: replacing the detailed meshes with simple convex shapes for collision, which would also
make the large files unnecessary and finalizing the MultiBodyPlant faster. 

Yahboom's meshes may carry redistribution terms separate from this package's Apache-2.0 license.
Check them before redistributing.

## Use

Build this package together with the bringup package that loads its model:

```bash
colcon build --packages-select x3plus_description x3plus_bringup
source install/setup.bash
```

`launch/rviz.launch.py` starts RViz with `rviz/yahboomcar.rviz`. It does not start
`robot_state_publisher` or publish joint states, so those must already be running for RViz to
show the robot.

The Drake pipeline does not use a launch file: the planning package loads the two Drake URDFs
directly, maps the mesh folder, and welds the gripper to `arm_link5`.

## Layout

```text
x3plus_description/
├── launch/rviz.launch.py
├── meshes/
│   ├── X3/                 # earlier X3 model assets
│   ├── X3plus/{collision,visual}/
│   └── sensor/{collision,visual}/
├── rviz/yahboomcar.rviz
├── urdf/
│   ├── x3plus.gazebo.urdf.xacro
│   ├── x3plus.gazebo.xacro
│   ├── x3plus_ros2_control.xacro
│   ├── drake_x3plus_fixed.urdf
│   ├── drake_x3plus_gripper.urdf
│   ├── yahboomcar_X3plus.urdf
│   └── yahboomcar_X3plus.urdf.xacro
├── CMakeLists.txt
├── LICENSE
└── package.xml
```

`CMakeLists.txt` installs `launch`, `meshes`, `urdf` and `rviz` into the package share directory.