import time
import numpy as np
import matplotlib.pyplot as plt


timesteps = int(1e6)  # us

microsteps = 32
steps_per_degree = 200/360

# setpoints
u_t1 = 20
target1 = int(2*microsteps*steps_per_degree)
u_t2 = 2e6
target2 = int(0*microsteps*steps_per_degree)

interval_counter = 0


# stepper state
moving = False
step = 0  # absolute position
current_speed = 0
current_step_interval = 0
direction = 1

# stepper profile
speed_interval_us = 20  # us
max_speed = int(1e6/speed_interval_us)  # us, minimum pulse time
print("max_speed :", max_speed)
min_speed = 100  # us, maximum pulse time
max_intervals = int(min_speed / speed_interval_us)
print("max_intervals :", max_intervals)
accel_rate = 100  # steps/s/s

_maxSpeed = 500

moving = True

# data arrays
time_list = np.zeros((timesteps,))
setpoint_list = np.zeros((timesteps,))
step_list = np.zeros((timesteps,))
vel_list = np.zeros((timesteps,))

for t in range(timesteps):

    setpoint = 0.0
    if t > u_t2:
        setpoint = target2
    elif t > u_t1:
        setpoint = target1

    # check speed
    # determine required speed
    dist = setpoint - step
    if dist > 0:
        required_speed = np.sqrt(2.0 * dist * accel_rate)
    elif dist < 0:
        required_speed = -np.sqrt(2.0 * -dist * accel_rate)
    else:
        required_speed = 0

    # control acceleration
    if (required_speed > current_speed):
        if (current_speed == 0):
            required_speed = np.sqrt(2.0 * accel_rate)
        else:
            required_speed = current_speed + abs(accel_rate / current_speed)

        if (required_speed > _maxSpeed):
            required_speed = _maxSpeed
    
    elif (required_speed < current_speed):
        if (current_speed == 0):
            required_speed = -np.sqrt(2.0 * accel_rate)
        else:
            required_speed = current_speed - abs(accel_rate / current_speed)

        if (required_speed < -_maxSpeed):
            required_speed = -_maxSpeed

    if (required_speed == 0):
        moving = False
    elif required_speed > 0:
        moving = True
        direction = 1
        current_speed = required_speed    
        current_step_interval = int( ( 1e6 / required_speed ) / speed_interval_us)

    elif required_speed < 0:
        moving = True
        direction = -1
        current_speed = -required_speed    
        current_step_interval = int( ( 1e6 / -required_speed) / speed_interval_us)

    # run the controler based on time interupt
    # determines if the stepper stepped or not
    if (t % speed_interval_us == 0):
        interval_counter +=1

        if (interval_counter >= current_step_interval) and moving:
            step += direction
            interval_counter = 0

    # if (interval_counter % max_intervals == 0):
    #     interval_counter = 0
    
    # print(dist, direction, current_speed, current_step_interval)
    # print(t, interval_counter)

    time_list[t] = t
    setpoint_list[t] = setpoint
    step_list[t] = step


setpoint_line, = plt.plot(time_list, setpoint_list, label="setpoint")
step_line, = plt.plot(time_list, step_list, label="y")
# vel_line, = plt.plot(time_list, vel_list, label="v")
# plt.legend(handles=[setpoint_line, state1_line])

plt.xlim((0, timesteps))
# plt.ylim((min(state1_list)-0.5, max(state1_list)+0.5))
plt.xlabel('steps')
plt.ylabel('value')
plt.legend()
plt.grid(True)
plt.show()
