import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
import math

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription_odometry = self.create_subscription(
            Odometry,
            '/odom',
            self.listener_callback_odometry,
            10)
        self.subscription_odometry  # prevent unused variable warning

        self.distance_total = 0
        self.prev_pos = None
        self.prev_ori = None
        self.battery_total = 0
        self.time_total = 0
        self.start_sec = None
        self.values_X = []
        self.values_Y = []
        self.values_alpha = []
        self.f_label_split = open("temp_coordinates.txt", "w")

    def listener_callback_odometry(self, msg):
        # print(msg.header)
        # print(msg.pose)
        # print(msg.pose.pose)
        # print(msg.pose.pose.position)
        # print(msg.pose.pose.orientation)
        current_pos=msg.pose.pose.position
        current_ori=msg.pose.pose.orientation
        if self.prev_pos is not None:
            this_distance = math.sqrt(pow(current_pos.x-self.prev_pos.x,2)+pow(current_pos.y-self.prev_pos.y,2)+pow(current_pos.z-self.prev_pos.z,2))
            self.distance_total+=this_distance
        print("Distance: %f m"%(self.distance_total))
        # print("AlphaW: %f"%(2*math.acos(current_ori.w)))
        print("AlphaZ: %f"%(2*math.asin(current_ori.z)))
        if self.distance_total>0.01:
            self.values_X.append(current_pos.x)
            self.values_Y.append(current_pos.y)
            self.values_alpha.append(2*math.asin(current_ori.z))
            self.f_label_split.write("%f %f %f %f\n"%(msg.header.stamp.sec,current_pos.x,current_pos.y,2*math.asin(current_ori.z)))
            if self.start_sec is None:
                self.start_sec = msg.header.stamp.sec
            self.time_total = msg.header.stamp.sec-self.start_sec
            print("Time: %f s"%(self.time_total))
        self.prev_pos=current_pos
        self.prev_ori=current_ori

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()