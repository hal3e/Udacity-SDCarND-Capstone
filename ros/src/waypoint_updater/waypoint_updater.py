#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

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

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.map_waypoints = None
        self.next_waypoint_index = None
        self.counter = -1
        self.len_waypoints = None

        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        if self.map_waypoints == None:
            pass
        else:
            self.counter += 1
            if self.counter % 2 == 0:
                cur_next_waypoint_index = self.next_waypoint(msg)
                if cur_next_waypoint_index != self.next_waypoint_index:
                    self.next_waypoint_index = cur_next_waypoint_index

                    lane = Lane()
                    lane.header.frame_id = '/world'
                    lane.header.stamp = rospy.Time(0)

                    # add new waypoints check if we went over the limit
                    add_points_limit = len(self.map_waypoints) - self.next_waypoint_index
                    if add_points_limit <  LOOKAHEAD_WPS:
                        lane.waypoints = self.map_waypoints[self.next_waypoint_index : self.next_waypoint_index + add_points_limit]
                        lane.waypoints = lane.waypoints + self.map_waypoints[0 : LOOKAHEAD_WPS - add_points_limit]
                    else:
                        lane.waypoints = self.map_waypoints[self.next_waypoint_index : self.next_waypoint_index + LOOKAHEAD_WPS]

                    self.final_waypoints_pub.publish(lane)

    def next_waypoint(self, cur_pos):
        closest_waypoint_index = self.closest_waypoint(cur_pos)
        cx = self.map_waypoints[closest_waypoint_index].pose.pose.position.x
        cy = self.map_waypoints[closest_waypoint_index].pose.pose.position.y

        x = cur_pos.pose.position.x
        y = cur_pos.pose.position.y

        heading = math.atan2(cx - x, cy - y)
        q = cur_pos.pose.orientation
        yaw =  tf.transformations.euler_from_quaternion([q.w, q.x, q.y, q.z], axes='sxyz')[2]

        angle = math.fabs(yaw - heading)
        angle = min(2 * math.pi - angle, angle)

        if angle > math.pi/4:
            closest_waypoint_index += 1
            if closest_waypoint_index > self.len_waypoints - 1:
                closest_waypoint_index = 0

        return closest_waypoint_index


    def closest_waypoint(self, cur_pos):
        closest_waypoint_index = -1
        closest_dist = None

        search_waypoints = []
        indexes = []

        if self.next_waypoint_index == None:
            search_waypoints = self.map_waypoints
            indexes = range(self.len_waypoints)
        # if we have already one next_waypoint search in it's neigbohood
        else:
            for i in  range(-5,6,1):
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

    def dist_p_w(self, cur_pos, waypoint):
        a = cur_pos.pose.position
        b = waypoint.pose.pose.position
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)


    def waypoints_cb(self, lane):
        self.map_waypoints = lane.waypoints;
        self.len_waypoints = len(self.map_waypoints)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
