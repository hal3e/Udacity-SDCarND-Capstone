#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 3
DISTANCE_THRESHOLD = 70

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)
        # sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # List of positions that correspond to the line to stop in front of for a given intersection
        self.stop_line_positions = self.config['stop_line_positions']
        self.current_stop_waypoint_index = None
        self.current_stop_position_index = None
        self.last_stop_position_index = None

        self.bridge = CvBridge()
        self.detector_is_ready = False
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()
        self.detector_is_ready = True

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        # self.use_ground_truth_state = True
        self.use_ground_truth_state = False
        self.image_counter = 0

        rospy.spin()


    def pose_cb(self, msg):
        self.pose = msg


    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints


    def traffic_cb(self, msg):
        self.lights = msg.lights


    def dist_pp(self, cur_pos, traffic_light):
        a = cur_pos.pose.position
        b_x = traffic_light[0]
        b_y = traffic_light[1]
        return math.sqrt((a.x - b_x)**2 + (a.y - b_y)**2)

    def close_to_traffic_light(self):
        # Check if we are close to a traffic light first

        closest_trafficlight = None
        for i,tfl in enumerate(self.stop_line_positions):
            distance_to_traffic_light = self.dist_pp(self.pose, tfl)
            if closest_trafficlight == None or distance_to_traffic_light < closest_trafficlight:
                closest_trafficlight = distance_to_traffic_light
                self.current_stop_position_index = i

        return closest_trafficlight < DISTANCE_THRESHOLD


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

        # Check if detector has loaded the graph and is ready
        if self.detector_is_ready:
            self.image_counter += 1

            # Check if our pose and map waypoints are initialized
            if self.pose != None and self.waypoints != None:

                # Check if we are close to a traffic light first
                if self.close_to_traffic_light() and self.image_counter % 2 == 0:

                    light_wp, state = self.process_traffic_lights()

                    '''
                    Publish upcoming red lights at camera frequency.
                    Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
                    of times till we start using it. Otherwise the previous stable state is
                    used.
                    '''
                    if self.state != state:
                        self.state_count = 0
                        self.state = state
                    elif self.state_count >= STATE_COUNT_THRESHOLD:
                        self.last_state = self.state
                        light_wp = light_wp if state == TrafficLight.RED or state == TrafficLight.YELLOW else -1
                        self.last_wp = light_wp
                        self.upcoming_red_light_pub.publish(Int32(light_wp))
                    else:
                        self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                    self.state_count += 1


    def get_closest_waypoint(self, stop_position):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        closest_waypoint_index = -1
        closest_dist = None

        for i,w in enumerate(self.waypoints):
            dist = self.dist_t_w(stop_position, w)
            if closest_dist == None or closest_dist > dist:
                closest_dist = dist
                closest_waypoint_index = i

        return closest_waypoint_index


    def dist_t_w(self, stop_position, waypoint):
        a_x = stop_position[0]
        a_y = stop_position[1]
        a_z = 0
        b = waypoint.pose.pose.position
        return math.sqrt((a_x-b.x)**2 + (a_y-b.y)**2  + (a_z-b.z)**2)


    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, desired_encoding="rgb8")

        # save_file_name = "./train-pictures/image_" + str(self.image_counter) + ".png"
        # self.image_counter += 1
        # cv2.imwrite(save_file_name, cv_image)

        #Get classification
        return self.light_classifier.get_classification(cv_image)


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # Check weather the stop position changed and update it accordingly
        if self.last_stop_position_index != self.current_stop_position_index:
            self.last_stop_position_index = self.current_stop_position_index
            self.current_stop_waypoint_index = self.get_closest_waypoint(self.stop_line_positions[self.current_stop_position_index])

        # Get current traffic light state
        state = self.get_light_state()

        if self.use_ground_truth_state:
            return self.current_stop_waypoint_index, self.lights[self.last_stop_position_index].state
        else:
            return self.current_stop_waypoint_index, state



if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
