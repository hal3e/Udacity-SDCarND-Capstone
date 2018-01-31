#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish
MAX_DECEL = 0.35


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_waypoints_cb, queue_size=1)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.traffic_waypoints_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.map_waypoints = None
        self.next_waypoint_index = None
        self.len_waypoints = None
        self.traffic_waypoint = -1
        self.last_traffic_waypoint = -1
        self.normal_velocity = None

        rospy.spin()


    def traffic_waypoints_cb(self, msg):
        self.traffic_waypoint = msg.data


    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x


    def set_waypoint_velocity_all(self, waypoints, velocity):
        for w in waypoints:
            w.twist.twist.linear.x = velocity


    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


    def waypoints_cb(self, lane):
        self.map_waypoints = lane.waypoints;
        self.len_waypoints = len(self.map_waypoints)

        # Get the desired velocity from the middle point
        self.normal_velocity = self.map_waypoints[int(self.len_waypoints/2)].twist.twist.linear.x


    def dist_p_w(self, cur_pos, waypoint):
        a = cur_pos.pose.position
        b = waypoint.pose.pose.position
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)


    def pose_cb(self, msg):
        if self.map_waypoints != None:

            # Get next waypoint from current position
            cur_next_waypoint_index = self.next_waypoint(msg)

            # Check whether we have already evaluated this potision, could happen when we are recieving the current position many times
            # or we are standing still. If the traffic light has changed then recalcualte
            if cur_next_waypoint_index != self.next_waypoint_index or self.last_traffic_waypoint != self.traffic_waypoint:

                # Save the next waypoint
                self.next_waypoint_index = cur_next_waypoint_index

                # Generate new lane message for final_waypoints
                lane = Lane()
                lane.header.frame_id = '/world'
                lane.header.stamp = rospy.Time(0)

                # Add new waypoints
                lane.waypoints = self.add_new_waypoints()

                # Procces traffic light info
                self.procces_traffic_light(lane.waypoints)

                # Publish the Lane message
                self.final_waypoints_pub.publish(lane)
                self.last_traffic_waypoint = self.traffic_waypoint


    def procces_traffic_light(self, lane_waypoints):
        # Get the traffic light index in the final waypoints index range
        traffic_light_index = self.traffic_waypoint - self.next_waypoint_index

        # Check the traffic light status and wether we are close to a stop waypoint or did we already pass it
        if self.traffic_waypoint == -1 or self.next_waypoint_index >= self.traffic_waypoint or traffic_light_index > LOOKAHEAD_WPS - 1:

            # If we changed from stop to go change the velocity to the default one
            if self.last_traffic_waypoint != self.traffic_waypoint:
                self.set_waypoint_velocity_all(lane_waypoints, self.normal_velocity)

        elif self.distance(self.map_waypoints, self.next_waypoint_index, self.traffic_waypoint) > 10.0:
                self.traffic_light_brake(lane_waypoints, traffic_light_index)


    def add_new_waypoints(self):
        new_waypoints = []

        # Add new waypoints check if we went over the limit map_waypoints size
        add_points_limit = self.len_waypoints - self.next_waypoint_index

        if add_points_limit <  LOOKAHEAD_WPS:
            new_waypoints = self.map_waypoints[self.next_waypoint_index : self.next_waypoint_index + add_points_limit]
            new_waypoints = new_waypoints + self.map_waypoints[:LOOKAHEAD_WPS - add_points_limit]
        else:
            new_waypoints = self.map_waypoints[self.next_waypoint_index : self.next_waypoint_index + LOOKAHEAD_WPS]

        return new_waypoints


    def traffic_light_brake(self, lane_waypoints, traffic_light_index):
        # Update the waypoint speed to decrease incrementally
        last = lane_waypoints[traffic_light_index]
        last.twist.twist.linear.x = 0.

        for i in range(len(lane_waypoints[:traffic_light_index])):
            dist = self.distance(lane_waypoints, i, traffic_light_index)
            vel = MAX_DECEL * dist

            if dist < 5.0:
                vel = 0.

            lane_waypoints[i].twist.twist.linear.x = min(vel, lane_waypoints[i].twist.twist.linear.x)

        # Set the rest of the waypoints to 0
        self.set_waypoint_velocity_all(lane_waypoints[traffic_light_index:], 0.0)


    def next_waypoint(self, cur_pos):
        # Get the next waypoint by first finding the closesnt one and then checking wether it is infront
        closest_waypoint_index = self.closest_waypoint(cur_pos)

        if self.check_heading(self.map_waypoints[closest_waypoint_index], cur_pos):
            closest_waypoint_index += 1
            if closest_waypoint_index > self.len_waypoints - 1:
                closest_waypoint_index = 0

        return closest_waypoint_index


    def check_heading(self, a, b):
        # Check wether a point is in front of the vehicle
        cx = a.pose.pose.position.x
        cy = a.pose.pose.position.y

        x = b.pose.position.x
        y = b.pose.position.x

        heading = math.atan2(cx - x, cy - y)
        q = b.pose.orientation
        yaw =  tf.transformations.euler_from_quaternion([q.w, q.x, q.y, q.z], axes='sxyz')[2]

        angle = math.fabs(yaw - heading)
        angle = min(2 * math.pi - angle, angle)

        return angle > math.pi/4


    def closest_waypoint(self, cur_pos):
        # Find closest waypoint from the map_waypoints
        closest_waypoint_index = -1
        closest_dist = None

        search_waypoints = []
        indexes = []

        if self.next_waypoint_index == None:
            search_waypoints = self.map_waypoints
            indexes = range(self.len_waypoints)

        # If we have already one next_waypoint search in its neighbourhood
        else:
            for i in  range(-25,26,1):
                index = i + self.next_waypoint_index
                if index < 0:
                    index = self.len_waypoints - 1 + index
                elif index > self.len_waypoints - 1:
                    index = index - self.len_waypoints

                search_waypoints.append(self.map_waypoints[index])
                indexes.append(index)

        for i,w in enumerate(search_waypoints):
            dist = self.dist_p_w(cur_pos, w)
            if closest_dist == None or closest_dist > dist:
                closest_dist = dist
                closest_waypoint_index = i

        return indexes[closest_waypoint_index]


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
