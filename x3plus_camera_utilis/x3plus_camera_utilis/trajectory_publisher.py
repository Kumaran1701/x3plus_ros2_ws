import numpy as np
import rclpy
from rclpy.node import Node
from x3plus_msgs.msg import ArmJoint
from x3plus_msgs.srv import RobotArmArray
 
WAYPOINT_DT = 0.1          # seconds between waypoints - START HERE, then tune
TRAJECTORY_FILE = "trajectory_export.npz"
# GRIPPER_RAW_ANGLE is no longer hardcoded - queried live from the robot at
# startup via the existing CurrentAngle service, so this script never
# commands an unintended gripper move (a hardcoded guess here was wrong
# once checked against your actual init value of 30, not 90).
 
 
def drake_rad_to_servo_deg(q_rad_5):
    """Inverse of the driver's own feedback conversion for arm_joint1-5:
    driver does rad = (deg - 90) * pi/180  =>  deg = rad*180/pi + 90"""
    return np.degrees(q_rad_5) + 90.0
 
 
class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')
 
        data = np.load(TRAJECTORY_FILE)
        self.times = data["times"]
        self.q = data["q"]  # shape (N, 5), radians, Drake convention
        if self.q.shape[1] != 5:
            raise ValueError(f"Expected 5 arm joints, got shape {self.q.shape}")
 
        self.pub = self.create_publisher(ArmJoint, 'TargetAngle', 10)
 
        # Query the CURRENT gripper angle live, rather than guessing a
        # value - avoids commanding an unintended gripper move on the
        # very first message.
        self.grip_client = self.create_client(RobotArmArray, 'CurrentAngle')
        self.get_logger().info("Waiting for CurrentAngle service...")
        self.grip_client.wait_for_service()
        future = self.grip_client.call_async(RobotArmArray.Request())
        rclpy.spin_until_future_complete(self, future)
        current_angles = future.result().angles
        self.gripper_raw_angle = current_angles[5]
        self.get_logger().info(f"Holding gripper fixed at current angle: {self.gripper_raw_angle}")
 
        self.index = 0
        self.run_time_ms = int(WAYPOINT_DT * 1000) + 20  # small overlap margin
                                                          # so each interpolation
                                                          # finishes just as the
                                                          # next command arrives
 
        self.get_logger().info(
            f"Loaded {len(self.times)} waypoints spanning {self.times[-1]:.2f}s. "
            f"Publishing every {WAYPOINT_DT}s with run_time={self.run_time_ms}ms."
        )
 
        self.timer = self.create_timer(WAYPOINT_DT, self.publish_next_waypoint)
 
    def publish_next_waypoint(self):
        if self.index >= len(self.times):
            self.get_logger().info("Trajectory complete.")
            self.timer.cancel()
            return
 
        q_rad = self.q[self.index]
        servo_deg = drake_rad_to_servo_deg(q_rad)
 
        msg = ArmJoint()
        # 6 values: 5 arm joints + gripper held fixed - matches self.joints
        # layout in the driver (arm_joint1..5, grip_joint)
        msg.joints = [float(v) for v in servo_deg] + [float(self.gripper_raw_angle)]
        msg.run_time = self.run_time_ms
 
        self.pub.publish(msg)
        self.get_logger().info(
            f"Waypoint {self.index}/{len(self.times)}  t={self.times[self.index]:.2f}s  "
            f"servo_deg={np.round(servo_deg, 1)}"
        )
        self.index += 1
 
 
def main():
    rclpy.init()
    node = TrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()