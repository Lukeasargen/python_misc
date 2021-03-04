import matplotlib.pyplot as plt
import numpy as np


def get_state_estimate(rrev, d_rrev, accel):
    pos = -(accel)/(rrev*rrev - d_rrev)
    vel = rrev*pos
    return [pos, vel]

def error(pred, true):
    return (pred-true)/(true)*100


if __name__ == "__main__":

    # Simulation setup
    timesteps = 1000000
    dt = 0.00001  # s

    # Inital conditions
    x0 = 0.1  # m from ceiling
    v0 = 1.41  # m/s towards ceiling
    a = -9.81  # m/s/s towards ceiling, change in velocity

    # Simulation variables
    x = x0
    v = v0
    rrev = v0/x0

    # Estimate collision time from inital conditions
    print("1/rrev =", 1/rrev)

    det = v0*v0 - 2*a*-x0
    if det <= 0:
        print("No collision")
    else:
        ttc = (-v0 + np.sqrt(det)) / (a)
        print("Time to collision (free-fall) = {:.6f} seconds.".format(ttc))
        print("Collision sim step = {}".format(int(ttc/dt)))

    # For graphing
    true_pos = []
    true_vel = []
    est_pos = []
    est_vel = []
    pos_err = []
    vel_err = []
    
    for t in range(timesteps):
        # step the simulation variables
        v = v + a*dt
        x = x - v*dt
        if x < 0 or v<-v0:
            print("Sim ended at {:.6f} seconds.".format(t*dt))
            t -=1  # correct for leaving the loop early
            break

        new_rrev = v/x
        d_rrev = (new_rrev-rrev)/dt
        rrev = new_rrev

        # estimate the state
        pos, vel = get_state_estimate(new_rrev, d_rrev, a)
        # print(x, v, rrev, pos, vel)

        # Store for graphing
        true_pos.append(x)
        true_vel.append(v)
        est_pos.append(pos)
        est_vel.append(vel)
        pos_err.append(error(pos, x))
        vel_err.append(error(vel, v))


    fig, ax = plt.subplots(2, 2)

    ax[0][0].set_xlabel('Timestep')
    ax[0][0].set_ylabel('Position')
    true_pos_line, = ax[0][0].plot(list(range(t+1)), true_pos, label="true_pos")
    est_pos_line, = ax[0][0].plot(list(range(t+1)), est_pos, label="est_pos")
    ax[0][0].legend(handles=[true_pos_line, est_pos_line])
    ax[0][0].set_xlim((0, t+1))
    ax[0][0].set_ylim((min(min(true_pos), min(est_pos)), max(max(true_pos), max(est_pos))))
    ax[0][0].set_title('Position')
    ax[0][0].grid(True)


    ax[0][1].set_xlabel('Timestep')
    ax[0][1].set_ylabel('Velocity')
    true_vel_line, = ax[0][1].plot(list(range(t+1)), true_vel, label="true_vel")
    est_vel_line, = ax[0][1].plot(list(range(t+1)), est_vel, label="est_vel")
    ax[0][1].legend(handles=[true_vel_line, est_vel_line])
    ax[0][1].set_xlim((0, t+1))
    ax[0][1].set_ylim((min(min(true_vel), min(est_vel))-0.1, max(max(true_vel), max(est_vel))+0.1))
    ax[0][1].set_title('Velocity')
    ax[0][1].grid(True)

    if det <= 0:
        scale = 1
    else:
        scale = 0.005 # inspect positions near 0 for collisions

    ax[1][0].set_xlabel('true_pos')
    ax[1][0].set_ylabel('Position Error')
    ax[1][0].plot(true_pos, pos_err, label="pos_err")
    ax[1][0].set_xlim((0, x0*scale))
    ax[1][0].set_ylim((min(pos_err), max(pos_err)))
    ax[1][0].set_title('Position Error')
    ax[1][0].grid(True)


    ax[1][1].set_xlabel('true_pos')
    ax[1][1].set_ylabel('Velocity Error')
    ax[1][1].plot(true_pos, vel_err, label="vel_err")
    ax[1][1].set_xlim((0, x0*scale))
    ax[1][1].set_ylim((min(vel_err), max(vel_err)))
    ax[1][1].set_title('Velocity Error')
    ax[1][1].grid(True)


    fig.tight_layout()
    plt.show()
