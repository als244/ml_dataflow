import numpy as np
import matplotlib
import sys

if sys.platform == 'darwin':
    matplotlib.use('MacOSX') # Using TkAgg backend (adjust if needed)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if len(sys.argv) != 3:
    print("Error:Usage: python transformer_torus.py <model_stages> <data_parallelism_factor>")
    sys.exit(1)

# --- Parameters ---
R_major = 4.0
r_minor = 1.0
N_slices = int(sys.argv[1])
M_nodes_per_slice = int(sys.argv[2])
twist_factor = 1 # <--- NEW: Integer twist factor (alpha). Use 1 for the simplest correction.

# --- Calculate Angles ---
phi_slices = np.linspace(0, 2 * np.pi, N_slices, endpoint=False)
theta_nodes_base = np.linspace(0, 2 * np.pi, M_nodes_per_slice, endpoint=False) # Base angles

# --- Calculate Node Coordinates ---
nodes = np.zeros((N_slices, M_nodes_per_slice, 3))
for k in range(N_slices):
    phi = phi_slices[k] # Toroidal angle for the current slice
    for j in range(M_nodes_per_slice):
        theta_base = theta_nodes_base[j] # Base poloidal angle for node index j

        # --- MODIFICATION START ---
        # Apply the twist: Adjust the poloidal angle based on the toroidal angle (phi)
        # This makes the node "rotate" around the minor radius as we move between slices.
        theta_twisted = theta_base + twist_factor * phi
        # The modulo operation is not strictly necessary as trig functions handle periodicity,
        # but can be useful for understanding the angle range.
        # theta_twisted = theta_twisted % (2 * np.pi)
        # --- MODIFICATION END ---

        # Calculate coordinates using the twisted theta
        radius_factor = R_major + r_minor * np.cos(theta_twisted) # Radius from center depends on twisted angle
        x = radius_factor * np.cos(phi)
        y = radius_factor * np.sin(phi)
        z = r_minor * np.sin(theta_twisted) # Height depends on twisted angle
        nodes[k, j, :] = [x, y, z]

# --- Define Colors ---
# Color definition remains the same, based on the initial node index j.
base_hues = np.linspace(0, 1, M_nodes_per_slice, endpoint=False)
# Optional: Keep your hue shift if you want a specific color (e.g., Red) to start at j=0
shift_numerator = 0.0 # Set to 0 to start with Red at j=0, or keep your value (e.g., 2.0)
shift_amount = shift_numerator / M_nodes_per_slice
shifted_hues = (base_hues - shift_amount) % 1.0
node_wire_colors = plt.cm.gist_ncar(shifted_hues)

# --- Create the 3D Plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# --- Create and Plot Shaded Poloidal Rings ---
# Note: These polygons connect nodes within the *same* slice k.
# With the twist, these nodes form a loop on the torus surface at angle phi_k.
verts = [nodes[k, :, :] for k in range(N_slices)]
ring_shading = Poly3DCollection(verts, facecolors='black', linewidths=0, edgecolors=None, alpha=0.3)
ax.add_collection3d(ring_shading)

# --- Plot Nodes and Wires ---
# This part correctly plots the helical paths by connecting nodes with the same index j across different slices k.
for j in range(M_nodes_per_slice):
    node_color = node_wire_colors[j]
    nodes_for_this_j = nodes[:, j, :] # Get all nodes for color j across all slices
    # Plot nodes
    ax.scatter(nodes_for_this_j[:, 0], nodes_for_this_j[:, 1], nodes_for_this_j[:, 2],
               color=node_color, s=75, depthshade=True, edgecolors='k', linewidth=0.5)
    # Plot wires connecting nodes of the same color (index j) between slices
    wire_points = np.vstack([nodes_for_this_j, nodes_for_this_j[0]]) # Close the loop
    ax.plot(wire_points[:, 0], wire_points[:, 1], wire_points[:, 2],
            color=node_color, linewidth=1.5, label=f"Seq. {j}")

# --- Add Labels to the Center of Each Face ---
# The centroid calculation still works to place labels near the center of each slice's node loop.
for k in range(N_slices):
    center_coords = np.mean(nodes[k, :, :], axis=0)
    ax.text(center_coords[0], center_coords[1], center_coords[2],
            str(k), color='darkgreen', ha='center', va='center',
            fontsize=9, fontweight='bold')

# --- Customize the Plot (as before) ---
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_title(f"Model Stages: {N_slices}, Data Parallelism Factor: {M_nodes_per_slice}")

# Set aspect ratio
x_min, x_max = nodes[:,:,0].min(), nodes[:,:,0].max()
y_min, y_max = nodes[:,:,1].min(), nodes[:,:,1].max()
z_min, z_max = nodes[:,:,2].min(), nodes[:,:,2].max()
buffer = 0.1 * (R_major + r_minor)
x_min -= buffer; x_max += buffer
y_min -= buffer; y_max += buffer
z_min -= buffer; z_max += buffer

max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0
mid_x = (x_max+x_min) * 0.5
mid_y = (y_max+y_min) * 0.5
mid_z = (z_max+z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.legend(title="")
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(False)

# --- Show or Save Plot ---
plt.show()
# plt.savefig("torus_twisted_rings.png", dpi=300)