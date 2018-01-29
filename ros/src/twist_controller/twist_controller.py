from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.vel_pid_ = PID(0.1, 0.0, 0.0, mn=0, mx=1)
        self.yaw_controller_ = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

    def control(self, linear_velocity, angular_velocity, current_velocity):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        throttle = self.vel_pid_.step(linear_velocity - current_velocity, 0.02)
        brake = 0.0
        if throttle < 0:
            brake = throttle

        steering = self.yaw_controller_.get_steering(linear_velocity, angular_velocity, current_velocity)

        return throttle, brake, steering
