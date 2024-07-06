import numpy as np
import pandas as pd
import mujoco.viewer
import matplotlib.pyplot as plt
import time

model = mujoco.MjModel.from_xml_path(filename='planar3_with_mass.xml')
data = mujoco.MjData(model)

# Define a linear trajectory (cross) for the end-effector
amplitude = 0.04  # 4 cm
max_velocity = 0.4  # 0.4 m/s
duration = 1.0  # duration in seconds
timestep = model.opt.timestep
num_steps = int(duration / timestep)

t = np.linspace(0, duration, num_steps)

# Generate cross in Cartesian space
x_traj = amplitude * np.sin(2 * np.pi * t / duration)
y_traj = amplitude * np.cos(2 * np.pi * t / duration)
z_traj = np.zeros_like(t)

initial_qpos = np.zeros(3)
results_dynamic = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(num_steps):
        data.qpos[:] = initial_qpos
        # Compute FK to get end-effector position
        mujoco.mj_forward(model, data)
        # Set target position for the end-effector
        target_pos = np.array([x_traj[i], y_traj[i], z_traj[i]])
        # Compute IK
        data.qpos[:] = target_pos
        # Calculate ID
        mujoco.mj_inverse(model, data)

        # Collect data
        results_dynamic.append([
            t[i],
            *data.qpos,
            *data.qvel,
            *data.qacc,
            *data.qfrc_inverse,
            *target_pos
        ])

        mujoco.mj_step(model, data)
        viewer.sync()

    start_time = time.time()
    while viewer.is_running() and time.time() - start_time < 10:
        viewer.sync()

columns_dynamic = ['time', 'q1', 'q2', 'q3', 'qvel1', 'qvel2', 'qvel3', 'qacc1', 'qacc2', 'qacc3', 'torque1', 'torque2',
                   'torque3', 'x', 'y', 'z']
df_dynamic = pd.DataFrame(results_dynamic, columns=columns_dynamic)
df_dynamic.to_csv('lec3_dynamic.csv', index=False)

fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

# Position
axs[0].plot(df_dynamic['time'], df_dynamic[['q1', 'q2', 'q3']])
axs[0].set_ylabel('Position (rad)')
axs[0].legend(['Joint 1', 'Joint 2', 'Joint 3'])
axs[0].set_title('Position vs Time')

# Velocity
axs[1].plot(df_dynamic['time'], df_dynamic[['qvel1', 'qvel2', 'qvel3']])
axs[1].set_ylabel('Velocity (rad/s)')
axs[1].legend(['Joint 1', 'Joint 2', 'Joint 3'])
axs[1].set_title('Velocity vs Time')

# Acceleration
axs[2].plot(df_dynamic['time'], df_dynamic[['qacc1', 'qacc2', 'qacc3']])
axs[2].set_ylabel('Acceleration (rad/s^2)')
axs[2].legend(['Joint 1', 'Joint 2', 'Joint 3'])
axs[2].set_title('Acceleration vs Time')

# Torque
axs[3].plot(df_dynamic['time'], df_dynamic[['torque1', 'torque2', 'torque3']])
axs[3].set_ylabel('Torque (Nm)')
axs[3].legend(['Joint 1', 'Joint 2', 'Joint 3'])
axs[3].set_title('Torque vs Time')
axs[3].set_xlabel('Time (s)')

plt.savefig('lec3_dynamic.png')
plt.show()
