# Required library: pip install pyvis
import os
import re
import webbrowser
from pyvis.network import Network
import colorsys


def hex_to_rgb(hex_color):
    """Converts a hex color string (e.g., '#FF0000') to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color format: {hex_color}")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """Converts an (R, G, B) tuple to a hex color string (e.g., '#FF0000')."""
    r, g, b = rgb_color
    # Clamp values to ensure they are within the valid 0-255 range
    r = max(0, min(255, int(round(r))))
    g = max(0, min(255, int(round(g))))
    b = max(0, min(255, int(round(b))))
    return f'#{r:02X}{g:02X}{b:02X}'

def generate_rainbow_shades(
    num_rainbow_base_colors=8,
    num_shades_per_base=16,
    base_saturation=1.0,   # Full saturation for vivid base colors (0.0 to 1.0)
    base_lightness=0.5,    # 50% lightness often gives most vivid hues (0.0 to 1.0)
    base_gray_hex='#808080', # Starting gray for the gray sequence
    target_shade_color_hex='#000000' # Target for darkening (Black)
):
    """
    Generates a dictionary of color shades including both a rainbow spectrum
    and a grayscale sequence.

    1. Creates `num_rainbow_base_colors` by taking even steps through the HUE
       component in HSL space.
    2. Creates a base gray color defined by `base_gray_hex`.
    3. For each base color (rainbow and gray), generates `num_shades_per_base`
       darkening shades by interpolating in RGB space towards `target_shade_color_hex`.

    Args:
        num_rainbow_base_colors (int): How many base colors for the hue spectrum.
        num_shades_per_base (int): How many darkening shades for EACH base color.
        base_saturation (float): Saturation for rainbow base colors (0.0 to 1.0).
        base_lightness (float): Lightness for rainbow base colors (0.0 to 1.0).
        base_gray_hex (str): The starting hex color for the gray sequence (e.g., '#808080').
        target_shade_color_hex (str): Color to interpolate towards for darkening.

    Returns:
        dict: A dictionary including keys like 'rainbow0_0', 'rainbowN_M',
              'gray_0', 'gray_M' and their hex color values.
    """
    if num_rainbow_base_colors < 0 or num_shades_per_base < 1:
        raise ValueError("Number of rainbow base colors must be non-negative, and shades must be at least 1.")
    if not (0.0 <= base_saturation <= 1.0 and 0.0 <= base_lightness <= 1.0):
         raise ValueError("Base saturation and lightness must be between 0.0 and 1.0.")

    generated_colors = {
        'red': '#FF0000', 'darkred': '#B22222', 'blue': '#0000FF', 'green': '#008000',
        'yellow': '#FFFF00', 'black': '#000000', 'white': '#FFFFFF',
        'gray': '#808080', 'lightgrey': '#D3D3D3', 'darkgreen': '#006400',
        'lightblue': '#ADD8E6', 'lightcoral': '#F08080',
        'lightgoldenrodyellow': '#FAFAD2', 'lightgreen': '#90EE90',
        'lightskyblue': '#87CEFA', 'mediumpurple': '#9370DB', 'purple': '#800080',
        'goldenrod': '#DAA520'
    }
    target_shade_rgb = hex_to_rgb(target_shade_color_hex)
    target_shade_r, target_shade_g, target_shade_b = target_shade_rgb

    # --- Part 1: Generate Rainbow Base Colors and Shades ---
    if num_rainbow_base_colors > 0:
        for j in range(num_rainbow_base_colors):
            current_hue = j / num_rainbow_base_colors
            current_base_rgb_float = colorsys.hls_to_rgb(current_hue, base_lightness, base_saturation)

            current_base_r = current_base_rgb_float[0] * 255
            current_base_g = current_base_rgb_float[1] * 255
            current_base_b = current_base_rgb_float[2] * 255
            base_name = f"col{j}"

            delta_shade_r = current_base_r - target_shade_r
            delta_shade_g = current_base_g - target_shade_g
            delta_shade_b = current_base_b - target_shade_b

            for i in range(num_shades_per_base):
                shade_factor = i / (num_shades_per_base - 1) if num_shades_per_base > 1 else 0
                current_shade_r = current_base_r - (delta_shade_r * shade_factor)
                current_shade_g = current_base_g - (delta_shade_g * shade_factor)
                current_shade_b = current_base_b - (delta_shade_b * shade_factor)
                current_shade_hex = rgb_to_hex((current_shade_r, current_shade_g, current_shade_b))
                shade_name = f"{base_name}_{i}"
                generated_colors[shade_name] = current_shade_hex

    # --- Part 2: Generate Grayscale Shades ---
    gray_base_name = 'gray'
    try:
        base_gray_rgb = hex_to_rgb(base_gray_hex)
    except ValueError:
        print(f"Warning: Invalid base_gray_hex '{base_gray_hex}'. Using default #808080.")
        base_gray_rgb = (128, 128, 128) # Default Gray RGB

    base_gray_r, base_gray_g, base_gray_b = base_gray_rgb
    delta_gray_r = base_gray_r - target_shade_r
    delta_gray_g = base_gray_g - target_shade_g
    delta_gray_b = base_gray_b - target_shade_b

    for i in range(num_shades_per_base):
        shade_factor = i / (num_shades_per_base - 1) if num_shades_per_base > 1 else 0
        current_shade_r = base_gray_r - (delta_gray_r * shade_factor)
        current_shade_g = base_gray_g - (delta_gray_g * shade_factor)
        current_shade_b = base_gray_b - (delta_gray_b * shade_factor)
        current_shade_hex = rgb_to_hex((current_shade_r, current_shade_g, current_shade_b))
        shade_name = f"{gray_base_name}_{i}"
        generated_colors[shade_name] = current_shade_hex

    return generated_colors


# --- NEW: Helper function to convert hex/named colors to RGBA ---
def hex_to_rgba(named_colors, color_input, alpha=1.0):
    """
    Converts a hex color string (#RRGGBB or #RGB) or a limited set of named
    colors to an RGBA string.

    Args:
        color_input (str): Hex color string (e.g., '#FF0000', '#F00') or a named color.
        alpha (float): Opacity level (0.0 to 1.0).

    Returns:
        str: RGBA string (e.g., 'rgba(255,0,0,0.8)') or the original input if conversion fails.
    """
    # Basic named color mapping (add more as needed)

    color_input = str(color_input).lower()
    hex_color = named_colors.get(color_input, color_input)

    # Clamp alpha
    alpha = max(0.0, min(1.0, alpha))

    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)

    try:
        if lv == 6: # RRGGBB
            rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        elif lv == 3: # RGB
            rgb = tuple(int(hex_color[i] * 2, 16) for i in (0, 1, 2))
        else:
            # If it's not a recognized hex or named color, return original
            # (pyvis might handle some named colors directly)
            print(f"Warning: Could not parse color '{color_input}'. Using it directly.")
            # Check if it already looks like rgba/rgb
            if color_input.startswith('rgb'):
                 return color_input # Assume it's already valid rgb/rgba
            return color_input # Fallback
        return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'
    except ValueError:
        print(f"Warning: Invalid hex color value '{hex_color}'. Using original '{color_input}'.")
        return color_input # Fallback

# Helper function to map graphviz shapes to pyvis shapes (approximate) - unchanged
def map_shape_gv_to_pyvis(gv_shape):
    """Maps common graphviz shape names to approximate pyvis shape names."""
    mapping = {
        'box': 'box',
        'ellipse': 'ellipse',
        'circle': 'circle',
        'invhouse': 'triangleDown', # Approximate
        'diamond': 'diamond',
        'Mdiamond': 'diamond',
        'octagon': 'hexagon',      # Approximate
        'doublecircle': 'dot',     # Use dot, maybe add border width later
        'folder': 'database',      # Approximate
        # Add more mappings as needed
    }
    #Ensure input is string and lowercase for matching
    gv_shape_str = str(gv_shape).lower()
    shape = mapping.get(gv_shape_str, 'dot') # Default to 'dot'

    # Special handling for doublecircle approximation
    border_width = 1 # Default border width
    if gv_shape_str == 'doublecircle':
        border_width = 4 # Make border thicker for doublecircle

    return shape, border_width


# Define the Pyvis-based visualizer class
class PyvisDependencyGraphVisualizer:
    def __init__(self, graph_name='DependencyGraph', initiaX_node_label="INPUT", num_base_colors=8, num_shades_per_base=16, **initiaX_node_attrs):
        """
        Initializes the interactive dependency graph using pyvis.

        Args:
            graph_name (str): The title heading for the graph page & browser tab.
            initiaX_node_label (str): The label for the starting node.
            **initiaX_node_attrs: Keyword arguments for styling the initial node
                                  (using graphviz-style names like fillcolor, shape,
                                   and NEW: opacity).
        """

        self.named_colors = generate_rainbow_shades(num_base_colors, num_shades_per_base)
        print(self.named_colors.keys())

        self.network = Network(
            height="800px",
            width="100%",
            directed=True,
            notebook=False,
            heading=graph_name
        )
        # --- CORRECTED OPTIONS STRING ---
        # Removed 'var options =' and all comments '/* ... */'
        self.network.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -250,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.4,
              "damping": 0.75,
              "avoidOverlap": 0.05
            },
            "minVelocity": 0.05,
            "solver": "forceAtlas2Based"
          },
          "nodes": {
            "font": {
                "strokeWidth": 2,
                "strokeColor": "#ffffff"
            }
          },
          "edges": {
            "font": {
                "align": "top",
                "background": "rgba(255, 255, 255, 0.7)",
                "strokeWidth": 0
            }
          }
        }
        """)
        # --- END OF CORRECTION ---

        # Add the initial node
        self.add_node(initiaX_node_label, **initiaX_node_attrs)
        self.initiaX_node_label = initiaX_node_label
        

    # ... (rest of the class methods remain the same) ...
    # _prepare_node_attributes, _prepare_edge_attributes, add_node,
    # add_node_with_dependencies, show, save


    def _prepare_node_attributes(self, default_attrs, **user_attrs):
        """
        Translates graphviz-style attributes (and opacity) to pyvis options and merges.
        """
        pyvis_attrs = {}
        # Merge default graphviz style with user-provided graphviz style
        # Prioritize user attributes
        merged_gv_attrs = {**default_attrs, **user_attrs}

        # --- NEW: Handle Opacity ---
        node_opacity = merged_gv_attrs.get('opacity', 1.0) # Default to opaque
        try:
            node_opacity = float(node_opacity)
        except (ValueError, TypeError):
            print(f"Warning: Invalid node opacity value '{merged_gv_attrs.get('opacity')}'. Using 1.0.")
            node_opacity = 1.0
        node_opacity = max(0.0, min(1.0, node_opacity)) # Clamp between 0 and 1

        # Translate fill color with opacity
        filX_color = merged_gv_attrs.get('fillcolor')
        if filX_color:
            rgba_filX_color = hex_to_rgba(self.named_colors, filX_color, node_opacity)
            # Use a dictionary for color to specify background
            pyvis_attrs['color'] = {
                'background': rgba_filX_color,
                # Keep border opaque unless specified otherwise (could add 'border_opacity')
                'border': hex_to_rgba(self.named_colors, merged_gv_attrs.get('color', 'black'), 1.0), # Default border black
                 # Make highlight/hover consistent with background opacity
                'highlight': {'background': rgba_filX_color, 'border': hex_to_rgba(self.named_colors, merged_gv_attrs.get('color', 'black'), 1.0)},
                'hover': {'background': rgba_filX_color, 'border': hex_to_rgba(self.named_colors, merged_gv_attrs.get('color', 'black'), 1.0)}
            }

        # Translate shape
        if 'shape' in merged_gv_attrs:
            shape, border_width = map_shape_gv_to_pyvis(merged_gv_attrs['shape'])
            pyvis_attrs['shape'] = shape
            pyvis_attrs['borderWidth'] = border_width
            # Ensure border color is set if borderWidth is > 0 and fillcolor wasn't set
            if 'color' not in pyvis_attrs and border_width > 0:
                 pyvis_attrs['color'] = {'border': hex_to_rgba(self.named_colors, merged_gv_attrs.get('color', 'black'), 1.0) }

        if 'borderwidth' in merged_gv_attrs: # Use a custom attribute name like 'borderwidth'
            try:
                pyvis_attrs['borderWidth'] = int(merged_gv_attrs['borderwidth'])
            except (ValueError, TypeError):
                print(f"Warning: Invalid borderwidth '{merged_gv_attrs['borderwidth']}'. Using default.")
                pyvis_attrs.setdefault('borderWidth', 1) # Set default if needed
        elif 'borderWidth' not in pyvis_attrs: # Ensure a default if not set by shape mapping or user
            pyvis_attrs['borderWidth'] = 1

        # Translate font size
        if 'fontsize' in merged_gv_attrs:
            try:
                pyvis_attrs.setdefault('font', {})['size'] = int(merged_gv_attrs['fontsize'])
            except (ValueError, TypeError):
                print(f"Warning: Invalid fontsize value '{merged_gv_attrs.get('fontsize')}'. Using default.")

        # --- Allow direct pyvis attributes ---
        for key, value in user_attrs.items():
            # Only add if not already handled by translation, avoids overwriting 'color' dict
            if key not in ['fillcolor', 'shape', 'fontsize', 'opacity', 'color']:
                 if key == 'font' and isinstance(value, dict) and 'font' in pyvis_attrs:
                     # Merge font dictionaries if both exist
                     pyvis_attrs['font'].update(value)
                 elif key not in pyvis_attrs: # Avoid overwriting things like shape
                    pyvis_attrs[key] = value

        # Set a default node size if not provided directly
        pyvis_attrs.setdefault('size', 25)

        return pyvis_attrs


    def _prepare_edge_attributes(self, default_attrs, **user_attrs):
        """
        Translates graphviz-style edge attributes (and opacity) to pyvis options and merges.
        """
        pyvis_attrs = {}
        # Merge default graphviz style with user-provided graphviz style
        merged_gv_attrs = {**default_attrs, **user_attrs}

        # --- NEW: Handle Opacity ---
        edge_opacity = merged_gv_attrs.get('opacity', 1.0) # Default to opaque
        try:
            edge_opacity = float(edge_opacity)
        except (ValueError, TypeError):
            print(f"Warning: Invalid edge opacity value '{merged_gv_attrs.get('opacity')}'. Using 1.0.")
            edge_opacity = 1.0
        edge_opacity = max(0.0, min(1.0, edge_opacity)) # Clamp between 0 and 1

        # Translate color with opacity
        edge_color_input = merged_gv_attrs.get('color')
        if edge_color_input:
            rgba_edge_color = hex_to_rgba(self.named_colors, edge_color_input, edge_opacity)
            # Apply opacity to both base and highlight color
            pyvis_attrs['color'] = {
                'color': rgba_edge_color,
                'highlight': rgba_edge_color,
                'hover': rgba_edge_color, # Keep hover consistent
                'inherit': False # Don't inherit color from nodes
            }
        elif edge_opacity != 1.0: # Apply default color with opacity if only opacity specified
             default_rgba_color = hex_to_rgba(self.named_colors, default_attrs.get('color', 'black'), edge_opacity)
             pyvis_attrs['color'] = {
                'color': default_rgba_color,
                'highlight': default_rgba_color,
                'hover': default_rgba_color,
                'inherit': False
            }


        # Translate penwidth
        if 'penwidth' in merged_gv_attrs:
            try:
                pyvis_attrs['width'] = float(merged_gv_attrs['penwidth'])
            except (ValueError, TypeError):
                print(f"Warning: Invalid penwidth value '{merged_gv_attrs.get('penwidth')}'. Using default.")
                pyvis_attrs.setdefault('width', 1)
        else:
            pyvis_attrs.setdefault('width', 1)

        # Translate style (dashed)
        if 'style' in merged_gv_attrs and str(merged_gv_attrs['style']).lower() == 'dashed':
            pyvis_attrs['dashes'] = True

        # --- Allow direct pyvis attributes ---
        for key, value in user_attrs.items():
             # Only add if not already handled by translation
             if key not in ['color', 'penwidth', 'style', 'opacity']:
                  pyvis_attrs[key] = value

        # Add default arrow styling if not specified
        pyvis_attrs.setdefault('arrows', 'to')

        return pyvis_attrs


    def add_node(self, node_label, **attributes):
        """
        Adds a single node to the graph with specified attributes.

        Args:
            node_label (str): The unique label/identifier for the node.
            **attributes: Keyword arguments for styling (e.g., fillcolor, shape,
                          fontsize, opacity). Can also include direct pyvis attributes
                          (e.g., size, title).
        """
        # Default graphviz-style attributes (opacity added here)
        default_node_attrs_gv = {'shape': 'ellipse', 'fillcolor': '#D3D3D3', 'opacity': 1.0}
        node_pyvis_options = self._prepare_node_attributes(default_node_attrs_gv, **attributes)

        # Add title (hover text) if not explicitly provided
        node_pyvis_options.setdefault('title', node_label)

        # Add the node
        self.network.add_node(node_label, label=node_label, **node_pyvis_options)
        print(f"Added Node: '{node_label}' with pyvis options: {node_pyvis_options}")


    def add_node_with_dependencies(self, node_label, dependencies, **node_attributes):
        """
        Adds a node and its outgoing dependencies (edges) to the graph.
        Edges point FROM the dependency TO the node being added (B -> A means A depends on B).

        Args:
            node_label (str): The label for the new node being added (A).
            dependencies (list): A list of tuples. Each tuple represents a prerequisite (B)
                                 and can be:
                - (edge_label, neighbor_node_label) -> Uses default edge style.
                - (edge_label, neighbor_node_label, edge_attrs_dict) -> Uses custom edge style
                  (using graphviz names like color, penwidth, style, and NEW: opacity).
            **node_attributes: Keyword arguments for styling the new node (A)
                               (e.g., fillcolor, shape, opacity).
        """
        # Add the node (A) itself first
        self.add_node(node_label, **node_attributes)

        # Add the edges (dependencies)
        if dependencies:
            print(f"Adding Edges TO '{node_label}':")
            # Default graphviz-style attributes for edges (opacity added here)
            default_edge_attrs_gv = {'color': 'black', 'penwidth': '1.0', 'opacity': 1.0}

            for dep_item in dependencies:
                edge_label = None
                neighbor_node_label = None # Prerequisite node (B)
                user_edge_attrs = {} # Graphviz style attributes + opacity

                if len(dep_item) == 3 and isinstance(dep_item[2], dict):
                    edge_label, neighbor_node_label, user_edge_attrs = dep_item
                elif len(dep_item) == 2:
                    edge_label, neighbor_node_label = dep_item
                else:
                    print(f"  - Warning: Skipping invalid dependency format: {dep_item}")
                    continue

                # Prepare pyvis options using the user attributes
                edge_pyvis_options = self._prepare_edge_attributes(default_edge_attrs_gv, **user_edge_attrs)

                # Add title (hover text) to edge if not explicitly provided
                edge_pyvis_options.setdefault('title', str(edge_label))
                # Set the visible label on the edge
                edge_pyvis_options['label'] = str(edge_label)


                # === Arrow Direction (FROM prerequisite B TO added node A) ===
                source_node = neighbor_node_label
                target_node = node_label
                self.network.add_edge(source_node, target_node, **edge_pyvis_options)
                # ===========================================================

                print(f"  - Edge: '{source_node}' --({edge_label})--> '{target_node}' with pyvis options: {edge_pyvis_options}")


    def show(self, filename='interactive_dependency_graph.html'):
        """Generates and attempts to open the interactive HTML graph file."""
        # (Code for saving/opening is unchanged)
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: '{output_dir}'")

        print(f"\nAttempting to generate interactive graph '{filename}'...")
        try:
            # Make sure to write the options before saving
            # self.network.set_options(self.network.options) # Ensure options are embedded
            self.network.save_graph(filename) # Save the graph first
            print(f"Interactive graph saved to '{filename}'.")
            try:
                file_uri = f'file://{os.path.realpath(filename)}'
                webbrowser.open(file_uri)
                print(f"Attempted to open '{file_uri}' in your default browser.")
            except Exception as e_open:
                print(f"Could not automatically open the file in browser: {e_open}")
        except Exception as e_save:
            print(f"Error generating interactive graph: {e_save}")

    def save(self, filename='interactive_dependency_graph.html'):
        """Generates and saves the interactive HTML graph file without opening."""
        # (Code for saving is unchanged)
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir, exist_ok=True)
             print(f"Created output directory: '{output_dir}'")

        print(f"\nSaving interactive graph to '{filename}'...")
        try:
             # self.network.set_options(self.network.options) # Ensure options are embedded
             self.network.save_graph(filename)
             print(f"Interactive graph saved successfully.")
        except Exception as e_save:
             print(f"Error saving interactive graph: {e_save}")


## USAGE!!!!

num_layers = 16


## deal with different sequence chunks proceeding through computation...
num_seq_chunks = 8


graph_viz = PyvisDependencyGraphVisualizer(
    graph_name="Transformer Dependency Graph for 1 Sequence Split into Chunks",
    initiaX_node_label="Raw Token Ids",
    fillcolor="black",
    shape="folder",
    fontsize=14,
    opacity=0.8, # Make initial node slightly transparent,
    num_base_colors = num_seq_chunks,
    num_shades_per_base = 2 * (num_layers + 3)
)

graph_viz.named_colors


# first add model weights

# add embedding layer
weight_node_size = 12
weight_node_shape = "box"
weight_node_fontsize = 48

graph_viz.add_node(f"W_E", 
                    fillcolor=f"gray_0",
                    color=f"darkgreen",
                    borderwidth='2',
                    shape=weight_node_shape,
                    size=weight_node_size,
                    fontsize=weight_node_fontsize)

# add blocks
for i in range(num_layers):
    graph_viz.add_node(f"W_{i}", 
                    fillcolor=f"gray_{2 * (i + 1)}",
                    color=f"darkgreen",
                    borderwidth='2',
                    shape=weight_node_shape,
                    size=weight_node_size,
                    fontsize=weight_node_fontsize)

# add head
graph_viz.add_node(f"Head", 
                    fillcolor=f"gray_{2 * (num_layers + 1)}",
                    shape=weight_node_shape,
                    size=weight_node_size,
                    fontsize=weight_node_fontsize)



embed_node_size = 12
embed_node_shape = "circle"
embed_node_fontsize = 36

block_node_size = 12
block_node_shape = "circle"

head_node_size = 12
head_node_shape = "circle"
head_node_fontsize=36

## FORWARDS DEPENDS

for i in range(num_seq_chunks):

    ## Embedding Layer
    graph_viz.add_node_with_dependencies(
        f"E_{i}",
        dependencies=[
            ("", "Raw Token Ids", {'color': f"col{i}_0", 'penwidth': '3.0'}),
            ("", "W_E",  {'color': f"gray_0", 'penwidth': '1.0'})
        ],
        fillcolor=f"col{i}_0",
        color=f"darkgreen",
        borderwidth='1',
        shape=embed_node_shape,
        opacity=0.85,
        size=embed_node_size,
        embed_node_fontsize=embed_node_fontsize
    )
    
    
    ## First Layer

    if i > 0:
        layer_depends = [
            ("Embedded Tokens", f"E_{i}", {'color': f"col{i}_1", 'penwidth': '3.0'}),
            ("Context", f"X_{i - 1},0", {'color': f'col{i - 1}_2', 'penwidth': '2.0', 'style': 'dashed'}),
            ("", "W_0",  {'color': f"gray_2", 'penwidth': '1.0'})
        ]
    else:
        layer_depends = [
            ("Embedded Tokens", f"E_{i}", {'color': f"col{i}_1", 'penwidth': '1.0'}),
            ("", "W_0",  {'color': f"gray_2", 'penwidth': '1.0'})
        ]

    graph_viz.add_node_with_dependencies(
        f"X_{i},0",
        dependencies=layer_depends,
        color=f"darkgreen",
        borderwidth='1',
        fillcolor=f"col{i}_2",
        shape=block_node_shape,
        opacity=0.85,
        size=block_node_size
    )
    
    ## Other Layers
    for j in range(1, num_layers):
        if i > 0:
            layer_depends=[("Activation Stream", f"X_{i},{j - 1}", {'color': f"col{i}_{2 * j + 1}", 'penwidth': '3.0'}),
                ("Context", f"X_{i - 1},{j}", {'color': f"col{i - 1}_{2 * j + 2}", 'penwidth': '2.0', 'style': 'dashed'}),
                ("", f"W_{j}",  {'color': f"gray_{2 * (j + 2)}", 'penwidth': '1.0'})]
        else:
            layer_depends=[("Activation Stream", f"X_{i},{j - 1}", {'color': f"col{i}_{2 * j + 1}", 'penwidth': '3.0'}),
            ("", f"W_{j}",  {'color': f"gray_{2 * (j + 2)}", 'penwidth': '1.0'})]
            
        
        graph_viz.add_node_with_dependencies(
            f"X_{i},{j}",
            color=f"darkgreen",
            borderwidth='1',
            dependencies=layer_depends,
            fillcolor=f"col{i}_{2 * j + 2}",
            shape=block_node_shape,
            size=block_node_size,
            opacity=0.85
        )

    ## head
    graph_viz.add_node_with_dependencies(
        f"H_{i}",
        dependencies=[
            ("Activation Stream", f"X_{i},{num_layers - 1}", {'color': f"col{i}_{2 * num_layers + 1}", 'penwidth': '3.0'}),
            ("", "Head",  {'color': f"gray_{2 * num_layers + 2}", 'penwidth': '1.0'})
        ],
        fillcolor=f"col{i}_{2 * num_layers + 2}",
        shape=head_node_shape,
        color="goldenrod",
        borderwidth='2',
        opacity=0.85,
        size=head_node_size,
        fontsize=head_node_fontsize
    )

    ## now do backwards...

## BACKWARDS DEPENDS

#dX stream!

grad_block_node_size = block_node_size
grad_block_node_shape = "circle"


## now start adding dependencies from final chunk
for i in range(num_seq_chunks - 1, -1, -1):


    ## ## last layer

    if i < num_seq_chunks - 1:
        grad_layer_depends = [
            ("Grad Stream", f"H_{i}", {'color': f"col{i}_{2 * (num_layers + 1) + 1}", 'penwidth': '3.0'}),
            ("Context Grad", f"dX_{i + 1},{num_layers - 1}", {'color': f'col{i + 1}_{2 * (num_layers + 1)}', 'penwidth': '2.0', 'style': 'dashed'}),
            ("Activations", f"X_{i},{num_layers - 1}", {'color': f"col{i}_{2 * (num_layers + 1)}", 'penwidth': '1.0'}),
            ("", f"W_{num_layers - 1}",  {'color': f"gray_{2 * (num_layers + 1)}", 'penwidth': '1.0'})
        ]
    else:
        grad_layer_depends = [
            ("Grad Stream", f"H_{i}", {'color': f"col{i}_{2 * (num_layers + 1) + 1}", 'penwidth': '3.0'}),
            ("Activations", f"X_{i},{num_layers - 1}", {'color': f"col{i}_{2 * (num_layers + 1)}", 'penwidth': '1.0'}),
            ("", f"W_{num_layers - 1}",  {'color': f"gray_{2 * (num_layers + 1)}", 'penwidth': '1.0'}),
        ]

    graph_viz.add_node_with_dependencies(
        f"dX_{i},{num_layers - 1}",
        color=f"darkred",
        borderwidth='1',
        dependencies=grad_layer_depends,
        fillcolor=f"col{i}_{2 * num_layers}",
        shape=grad_block_node_shape,
        opacity=0.85,
        size=grad_block_node_size
    )
    
    
    ## Other Layers
    for j in range(num_layers - 2, -1, -1):
        
        if i < num_seq_chunks - 1:
            grad_layer_depends=[("Grad Stream", f"dX_{i},{j + 1}", {'color': f"col{i}_{2 * (j + 1) + 1}", 'penwidth': '3.0'}),
                ("Context Grad", f"dX_{i + 1},{j}", {'color': f"col{i + 1}_{2 * (j + 2)}", 'penwidth': '2.0', 'style': 'dashed'}),
                ("Activations", f"X_{i},{j}", {'color': f"col{i}_{2 * (j + 1) + 1}", 'penwidth': '1.0'}),
                ("", f"W_{j}",  {'color': f"gray_{2 * (j + 1)}", 'penwidth': '1.0'})]
        else:
            grad_layer_depends=[("Grad Stream", f"dX_{i},{j + 1}", {'color': f"col{i}_{2 * (j + 1) + 1}", 'penwidth': '3.0'}),
                ("Activations", f"X_{i},{j}", {'color': f"col{i}_{2 * (j + 1) + 1}", 'penwidth': '1.0'}),
                ("", f"W_{j}",  {'color': f"gray_{2 * (j + 1)}", 'penwidth': '1.0'})]
        
        print(f"ADDING NODE: dX_{i},{j}\n\n\n\n")
        
        graph_viz.add_node_with_dependencies(
            f"dX_{i},{j}",
            color=f"darkred",
            borderwidth='1',
            dependencies=grad_layer_depends,
            fillcolor=f"col{i}_{2 * j + 2}",
            shape=grad_block_node_shape,
            size=grad_block_node_size,
            opacity=0.85,
        )






## dW stream!

grad_head_node_size = weight_node_size
grad_head_node_shape = weight_node_shape
grad_head_node_fontsize = weight_node_fontsize

## also need to add edges connecting all L{i},{j}, dX_{i},{j]} to dW_{j}

grad_head_node_depends = []

for i in range(num_seq_chunks):
    grad_head_node_depends += [("", f"H_{i}", {'color': f"col{i}_{2 * (num_layers + 1)}", 'penwidth': '1.0'})]


graph_viz.add_node_with_dependencies(
        f"dW_Head",
        color=f"darkred",
        borderwidth='1',
        dependencies=grad_head_node_depends,
        fillcolor=f"col{i}_{2 * (num_layers + 1)}",
        shape=grad_head_node_shape,
        opacity=0.85,
        size=grad_head_node_size,
        fontsize=grad_head_node_fontsize
    )


grad_block_weight_node_size = block_node_size
grad_block_weight_node_shape = block_node_shape

grad_weight_node_size = weight_node_size
grad_weight_node_shape = weight_node_shape
grad_weight_node_fontsize = weight_node_fontsize

# add blocks
for j in range(num_layers):

    grad_weight_depends = []

    for i in range(num_seq_chunks):
        grad_block_weight_depends = [("Activations", f"X_{i},{j}", {'color': f"col{i}_{2 * (j + 1)}", 'penwidth': '1.0'}),
                        ("Activations Grad", f"dX_{i},{j}", {'color': f"col{i}_{2 * (j + 1)}", 'penwidth': '1.0'})]


        graph_viz.add_node_with_dependencies(f"dW_{i},{j}", 
                    fillcolor=f"col{i}_{2 * (j + 1)}",
                    dependencies=grad_block_weight_depends,
                    shape=grad_block_weight_node_shape,
                    size=grad_block_weight_node_size,
                    opacity=.5)

        grad_weight_depends += [("", f"dW_{i},{j}", {'color': f"col{i}_{2 * (j + 1)}", 'penwidth': '1.0'})]

    
    ## add weight deriv
    graph_viz.add_node_with_dependencies(f"dW_{j}", 
                    fillcolor=f"gray_{2 * (j + 1)}",
                    color=f"darkred",
                    borderwidth='2',
                    dependencies=grad_weight_depends,
                    shape=grad_weight_node_shape,
                    size=grad_weight_node_size,
                    fontsize=grad_weight_node_fontsize)




# add embedding layer
grad_embed_depends = []
for i in range(num_seq_chunks):
    # cur_depends = [("", "Raw Token Ids", {'color': f"col{i}_0", 'penwidth': '2.0'}),
    #                ("", f"dX_{i},0", {'color': f"col{i}_1", 'penwidth': '2.0', 'opacity': 0.75})]

    cur_depends = [("Grad Stream", f"dX_{i},0", {'color': f"col{i}_1", 'penwidth': '3.0'})]
    grad_embed_depends += cur_depends



print(f"\n\n\n\n\n\n\n\nGRAD EMBED DEPENDS: {grad_embed_depends}\n\n\n\n\n\n")

graph_viz.add_node_with_dependencies(f"dW_E", 
                    fillcolor=f"gray_0",
                    color=f"darkred",
                    borderwidth='2',
                    dependencies=grad_embed_depends,
                    shape=grad_weight_node_shape,
                    size=grad_weight_node_size,
                    fontsize=grad_weight_node_fontsize)








# Show/Save the graph
output_filename = 'transformer_dependency_graph_seq_chunks.html'
graph_viz.show(output_filename)
#graph_viz.save(output_filename)