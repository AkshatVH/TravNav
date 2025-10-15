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
            '/panther/odometry/filtered',
            self.listener_callback_odometry,
            10)
        self.subscription_odometry  # prevent unused variable warning

        self.subscription_battery = self.create_subscription(
            BatteryState,
            '/panther/battery',
            self.listener_callback_battery,
            10)
        self.subscription_battery  # prevent unused variable warning

        self.distance_total = 0
        self.prev_pos = None
        self.prev_ori = None
        self.battery_total = 0
        self.time_total = 0
        self.start_sec = None

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
        self.prev_pos=current_pos
        self.prev_ori=current_ori

    def listener_callback_battery(self, msg):
        # print(msg.current)
        if self.distance_total>0.01:
            self.battery_total+=abs(msg.current)
            if self.start_sec is None:
                self.start_sec = msg.header.stamp.sec
            self.time_total = msg.header.stamp.sec-self.start_sec
            print("Time: %f s"%(self.time_total))
            print("Current: %f A"%(self.battery_total))
            if self.time_total>0:
                print("Current/sec: %f A/s"%(self.battery_total/self.time_total))


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