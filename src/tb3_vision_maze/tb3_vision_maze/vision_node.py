import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            '/maze_state',
            10
        )

        self.get_logger().info("Vision node checking for RED EXIT")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Convert to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Red color range (Two ranges for Red in HSV: 0-10 and 170-180)
        # S and V should be high
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        full_mask = mask1 + mask2
        
        # Count red pixels
        red_pixels = cv2.countNonZero(full_mask)
        
        # Threshold: Assuming image is 640x480 (307,200 pixels)
        # If exit is close, it takes up a significant portion.
        # We want to stop when we are CLOSE (e.g. < 0.5m).
        # At 0.5m, the 1.2m wide box fills the view. 
        # So threshold should be high.
        threshold = 100000
        
        out = String()
        if red_pixels > threshold:
            out.data = "exit_visible"
        else:
            out.data = "searching"
            
        self.publisher.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
