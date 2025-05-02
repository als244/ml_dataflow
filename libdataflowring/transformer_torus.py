import numpy as np
import matplotlib
matplotlib.use('MacOSX') # Using TkAgg backend (adjust if needed)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Parameters ---
R_major = 4.0
r_minor = 1.0
N_slices = 16 # Keep this number moderate for label visibility, e.g., 12-24
M_nodes_per_slice = 8

# --- Calculate Angles ---
phi_slices = np.linspace(0, 2 * np.pi, N_slices, endpoint=False)
theta_nodes = np.linspace(0, 2 * np.pi, M_nodes_per_slice, endpoint=False)

# --- Calculate Node Coordinates ---
nodes = np.zeros((N_slices, M_nodes_per_slice, 3))
for k in range(N_slices):
    for j in range(M_nodes_per_slice):
        phi = phi_slices[k]
        theta = theta_nodes[j]
        x = (R_major + r_minor * np.cos(theta)) * np.cos(phi)
        y = (R_major + r_minor * np.cos(theta)) * np.sin(phi)
        z = r_minor * np.sin(theta)
        nodes[k, j, :] = [x, y, z]

# --- Define Colors ---
# Calculate base hue values for M nodes (0 to almost 1, evenly spaced)
# endpoint=False ensures the start and end hues (0 and 1 for hsv) aren't identical
base_hues = np.linspace(0, 1, M_nodes_per_slice, endpoint=False)

# Shift the hues: We want the node at index j=2 (original hue 2/M) to get hue 0 (Red).
# So, we subtract 2/M from all hues. Use modulo 1 for correct wrapping.
shift_numerator = 2.0 # <--- CHANGE THIS VALUE
shift_amount = shift_numerator / M_nodes_per_slice
shifted_hues = (base_hues - shift_amount) % 1.0

# Generate colors using the SHIFTED hues and the chosen colormap
# Option 1: HSV (Standard, but start/end colors might appear similar)
node_wire_colors = plt.cm.gist_ncar(shifted_hues)

# --- Create the 3D Plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# --- Create and Plot Shaded Poloidal Rings ---
verts = [nodes[k, :, :] for k in range(N_slices)]
ring_shading = Poly3DCollection(verts, facecolors='black', linewidths=0, edgecolors=None, alpha=0.3)
ax.add_collection3d(ring_shading)

# --- Plot Nodes and Wires ---
for j in range(M_nodes_per_slice):
    node_color = node_wire_colors[j]
    nodes_for_this_j = nodes[:, j, :]
    # Plot nodes
    ax.scatter(nodes_for_this_j[:, 0], nodes_for_this_j[:, 1], nodes_for_this_j[:, 2],
               color=node_color, s=75, depthshade=True) # Example: increased node size to 100
    # Plot wires
    wire_points = np.vstack([nodes_for_this_j, nodes_for_this_j[0]])
    ax.plot(wire_points[:, 0], wire_points[:, 1], wire_points[:, 2],
            color=node_color, linewidth=1.5, label=f"Seq. {j}")

# --- **NEW**: Add Labels to the Center of Each Face ---
for k in range(N_slices):
    # Calculate the center (centroid) of the k-th face
    # This is the average of the coordinates of its nodes
    center_coords = np.mean(nodes[k, :, :], axis=0) # Average along the nodes axis

    # Add the text label (face number k) at the center coordinates
    ax.text(center_coords[0], center_coords[1], center_coords[2],
            str(k), # The text is the face number (0, 1, 2, ...)
            color='darkgreen', # Desired text color
            ha='center',       # Horizontal alignment
            va='center',       # Vertical alignment
            fontsize=9,        # Adjust font size as needed
            fontweight='bold') # Make it bold for better visibility
# --- End of New Section ---


# --- Customize the Plot (as before) ---
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
ax.set_title(f"Torus with Labeled Faces ({N_slices} Slices, {M_nodes_per_slice} Nodes/Slice)")

# Set aspect ratio
# ... (rest of aspect ratio calculation and setting code) ...
x_min, x_max = nodes[:,:,0].min(), nodes[:,:,0].max()
y_min, y_max = nodes[:,:,1].min(), nodes[:,:,1].max()
z_min, z_max = nodes[:,:,2].min(), nodes[:,:,2].max()
# Adjust buffer if labels go outside plot bounds initially
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


# Optional: Background and grid

ax.legend(title="Data Parallelism")

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.axis('off')
ax.grid(False)

# --- Show or Save Plot ---
plt.show()
# plt.savefig("torus_labeled_faces.png", dpi=300)
