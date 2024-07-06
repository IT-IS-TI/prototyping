import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import time

# Initialize the model and data
model = mujoco.MjModel.from_xml_path(filename='planar3_with_mass.xml')
data = mujoco.MjData(model)

# Define ranges for joint angles (in radians)
joint_ranges = [np.linspace(-np.pi, np.pi, num=10) for _ in range(3)]

# Initialize list to store results
results = []

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    for q1 in joint_ranges[0]:
        for q2 in joint_ranges[1]:
            for q3 in joint_ranges[2]:
                # Set joint angles
                data.qpos[:] = [q1, q2, q3]
                # Calculate inverse dynamics
                mujoco.mj_inverse(model, data)
                # Store the results
                results.append([q1, q2, q3] + list(data.qfrc_inverse))
                # Step the simulation to update the viewer
                mujoco.mj_step(model, data)
                viewer.sync()

    # Keep the viewer open for a few more seconds to view the final configuration
    start_time = time.time()
    while viewer.is_running() and time.time() - start_time < 10:
        viewer.sync()

# Create a DataFrame and save to CSV
df_static = pd.DataFrame(results, columns=['q1', 'q2', 'q3', 'torque1', 'torque2', 'torque3'])
df_static.to_csv('lec2_static_id_results.csv', index=False)

# Plotting the violin plot with Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
ax.violinplot([df_static['torque1'], df_static['torque2'], df_static['torque3']])
ax.set_xlabel('Joint')
ax.set_ylabel('Torque')
ax.set_title('Distribution of Torques in the Joints')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Joint 1', 'Joint 2', 'Joint 3'])
plt.savefig('lec2_static_torques.png')
plt.show()
