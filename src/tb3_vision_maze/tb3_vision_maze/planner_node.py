import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        self.vision_state = "searching"
        self.sub = self.create_subscription(
            String,
            '/maze_state',
            self.state_callback,
            10
        )
        self.scan_sub = self.create_subscription(
            String,
            '/scan_state',
            self.scan_callback,
            10
        )
        self.pub = self.create_publisher(
            String,
            '/planner_cmd',
            10
        )
        self.get_logger().info('Planner node started')

    def state_callback(self, msg):
        self.vision_state = msg.data

    def scan_callback(self, msg):
        # Scan state format: "L1_F1_R1"
        scan_data = msg.data
        
        action = 'STOP'
        
        # Priority 1: Check Vision Stop
        if self.vision_state == 'exit_visible':
            self.get_logger().info('EXIT FOUND! STOPPING.')
            action = 'STOP'
        else:
            # Priority 2: Left Hand Rule (Continuous)
            # Delegate complex wall following to Motion Node
            action = 'FOLLOW_LEFT'

        # frequency limit logging?
        # self.get_logger().info(f'Scan: {scan_data} | Vision: {self.vision_state} -> Action: {action}')
        self.pub.publish(String(data=action))

def main():
    rclpy.init()
    node = PlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()
