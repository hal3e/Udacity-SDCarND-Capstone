from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, v_mass, d_accel, wheel_radius):
        # PID controller for velocity
        self.vel_pid_ = PID(0.1, 0.0001, 0.0, mn=-2, mx=1)

        # Yaw controller buit by Udacity
        self.yaw_controller_ = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        self.max_decel = v_mass * (-d_accel) * wheel_radius
        self.k_decel = 1750

    def control(self, linear_velocity, angular_velocity, current_velocity):
        # Update throttle and brake command
        throttle = self.vel_pid_.step(linear_velocity - current_velocity, 0.0667)
        brake = 0.0

        # If throttle lower than 0.05 don't send any command, or if negative apply brake
        if throttle < 0.05:
            if throttle < 0.0:
                brake = abs(self.k_decel * throttle)

                if brake > self.max_decel:
                    brake = self.max_decel

            throttle = 0.0

        # Update steering command
        steering = self.yaw_controller_.get_steering(linear_velocity, angular_velocity, current_velocity)

        return throttle, brake, steering

    def reset(self):
        self.vel_pid_.reset()
