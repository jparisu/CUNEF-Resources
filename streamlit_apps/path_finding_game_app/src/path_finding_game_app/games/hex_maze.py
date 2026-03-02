import streamlit as st
import networkx as nx
import random
import math
import time

# --- CONSTANTS ---
HEX_SIZE = 25  # Pixel radius
SQRT3 = math.sqrt(3)

# --- HEX GRID LOGIC ---
# Axial coordinates: (q, r). Third coord s = -q-r

def hex_to_pixel(q, r):
    x = HEX_SIZE * (3/2 * q)
    y = HEX_SIZE * (SQRT3/2 * q + SQRT3 * r)
    return x, y

def get_neighbors(q, r):
    # 6 directions in axial coords
    directions = [
        (+1, 0), (+1, -1), (0, -1),
        (-1, 0), (-1, +1), (0, +1)
    ]
    return [(q + dq, r + dr) for dq, dr in directions]

def get_hex_direction(current, target):
    cq, cr = current
    tq, tr = target
    dq, dr = tq - cq, tr - cr

    mapping = {
        (1, 0): "SE", (1, -1): "NE", (0, -1): "N",
        (-1, 0): "NW", (-1, 1): "SW", (0, 1): "S"
    }
    return mapping.get((dq, dr), "?")

class HexMazeGenerator:
    @staticmethod
    def generate(radius, seed, num_starts, num_ends):
        random.seed(seed)

        # 1. Build Full Graph (Grid)
        G = nx.Graph()
        # Generate hexes in a spiral/circle
        nodes = []
        for q in range(-radius, radius + 1):
            r1 = max(-radius, -q - radius)
            r2 = min(radius, -q + radius)
            for r in range(r1, r2 + 1):
                nodes.append((q, r))
                G.add_node((q, r))

        # Add edges between neighbors if they exist
        for node in nodes:
            for neighbor in get_neighbors(*node):
                if neighbor in nodes:
                    # Add random weight for MST generation
                    G.add_edge(node, neighbor, weight=random.random())

        # 2. Maze Generation (MST = Perfect Maze)
        # Minimum Spanning Tree creates a graph with no loops (tree) covering all nodes
        maze_graph = nx.minimum_spanning_tree(G)

        # 3. Identify Edge Nodes (Boundary)
        boundary_nodes = []
        for node in nodes:
            # A node is on boundary if it has fewer than 6 neighbors in the original full grid
            # OR logic check: max coord absolute value == radius?
            # Axial distance logic: (abs(q) + abs(q+r) + abs(r)) / 2 == radius
            if (abs(node[0]) + abs(node[0] + node[1]) + abs(node[1])) // 2 == radius:
                boundary_nodes.append(node)

        if len(boundary_nodes) < num_starts + num_ends:
            # Fallback if map too small
            boundary_nodes = nodes

        random.shuffle(boundary_nodes)
        starts = boundary_nodes[:num_starts]
        ends = boundary_nodes[num_starts:num_starts+num_ends]

        return maze_graph, starts, ends

# --- SVG RENDERER ---

def generate_svg(maze_graph, player_pos, starts, ends, solution_path=None):
    # Calculate bounds
    all_nodes = list(maze_graph.nodes())
    xs = []
    ys = []
    for q, r in all_nodes:
        x, y = hex_to_pixel(q, r)
        xs.append(x)
        ys.append(y)

    pad = HEX_SIZE + 5
    min_x, max_x = min(xs) - pad, max(xs) + pad
    min_y, max_y = min(ys) - pad, max(ys) + pad
    width = max_x - min_x
    height = max_y - min_y

    svg = [f'<svg viewBox="{min_x} {min_y} {width} {height}" width="600" height="500" xmlns="http://www.w3.org/2000/svg">']

    # CSS Styles within SVG
    svg.append('''
    <style>
        .hex { fill: #f0f2f6; stroke: #bcc6d4; stroke-width: 1; }
        .wall { stroke: #2c3e50; stroke-width: 3; stroke-linecap: round; }
        .start { fill: #27ae60; fill-opacity: 0.5; }
        .end { fill: #c0392b; fill-opacity: 0.5; }
        .player { fill: #f1c40f; stroke: #f39c12; stroke-width: 2; }
        .path { fill: none; stroke: #3498db; stroke-width: 4; stroke-dasharray: 5,5; opacity: 0.6; }
    </style>
    ''')

    # Draw Solution Path (Underneath)
    if solution_path:
        pts = []
        for node in solution_path:
            x, y = hex_to_pixel(*node)
            pts.append(f"{x},{y}")
        polyline = f'<polyline points="{" ".join(pts)}" class="path" />'
        svg.append(polyline)

    # Draw Hexes (Cells)
    for q, r in all_nodes:
        cx, cy = hex_to_pixel(q, r)

        # Hexagon Points
        points = []
        for i in range(6):
            angle_deg = 60 * i + 30
            angle_rad = math.pi / 180 * angle_deg
            px = cx + HEX_SIZE * math.cos(angle_rad)
            py = cy + HEX_SIZE * math.sin(angle_rad)
            points.append(f"{px:.1f},{py:.1f}")

        cls = "hex"
        if (q, r) in starts: cls += " start"
        if (q, r) in ends: cls += " end"

        svg.append(f'<polygon points="{" ".join(points)}" class="{cls}" />')

    # Draw Walls (Missing Edges)
    # We check all 6 potential neighbors. If edge not in maze_graph, draw wall.
    # To avoid drawing walls twice, we can establish an order, or just draw all and overlap.
    for q, r in all_nodes:
        cx, cy = hex_to_pixel(q, r)
        neighbors = get_neighbors(q, r)

        for i, (nq, nr) in enumerate(neighbors):
            # Calculate the shared edge segment
            angle1_deg = 60 * i + 30 - 30 # Corner before
            angle2_deg = 60 * i + 30 + 30 # Corner after
            # Wait, points are at +30, +90, +150...
            # Neighbor 0 is (+1, 0) -> East?
            # Let's rely on geometry: Midpoint between centers is the wall center.
            # Wall is perpendicular to vector (neighbor - current).

            # Simple approach: Check graph connectivity
            has_path = maze_graph.has_edge((q, r), (nq, nr))

            if not has_path:
                # Wall exists. Draw line between the two shared vertices.
                # Vertices for direction i are i and (i+1)%6 ?
                # 0: (+1, 0) corresponds to vertices at angle -30 and +30?
                # Actually, simpler:
                # The vertices of the hex are at angles 30, 90, 150, 210, 270, 330.
                # The edge towards neighbor i connects vertex i and (i+1)%6?
                # Let's align directions with vertices.
                # Directons: 0=(1,0), 1=(1,-1), 2=(0,-1)...
                # 0 is E. Vertices 30(SE) and 330(NE) form the East edge? No, 330 and 30.
                # Let's hardcode vertex pairs for neighbors 0..5
                # 0 (East, +1,0): Vertices at 330 (-30) and 30.
                # 1 (NE, +1,-1): Vertices at 270 and 330.

                # Mapping neighbor index to vertex angles
                # Neighbors list: (+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)
                # These are roughly: E, NE, NW, W, SW, SE ? No.
                # Axial coords are skewed.
                # Let's just use midpoints.
                pass

    # Re-drawing walls: Easier way.
    # For each node, for each neighbor direction:
    # If neighbor exists in grid but NO edge in graph -> Draw Wall.
    # Wall is a line segment.

    # Vertices of a hex at (cx, cy):
    # v0: 30deg (SE approx), v1: 90deg (S), v2: 150deg (SW), v3: 210 (NW), v4: 270 (N), v5: 330 (NE)
    # Directions:
    # 0 (+1, 0): Right. Edge between v5 and v0.
    # 1 (+1, -1): Top Right. Edge between v4 and v5.
    # 2 (0, -1): Top Left. Edge between v3 and v4.
    # 3 (-1, 0): Left. Edge between v2 and v3.
    # 4 (-1, 1): Bottom Left. Edge between v1 and v2.
    # 5 (0, 1): Bottom Right. Edge between v0 and v1.

    vertex_angles = [30, 90, 150, 210, 270, 330]

    for q, r in all_nodes:
        cx, cy = hex_to_pixel(q, r)
        neighbors = get_neighbors(q, r)

        # Directions corresponding to neighbors list in get_neighbors
        # (+1, 0) is Right -> Edge v5-v0
        # (+1, -1) is Top Right -> Edge v4-v5
        # (0, -1) is Top Left -> Edge v3-v4
        # (-1, 0) is Left -> Edge v2-v3
        # (-1, +1) is Bot Left -> Edge v1-v2
        # (0, +1) is Bot Right -> Edge v0-v1

        edge_pairs = [(5, 0), (4, 5), (3, 4), (2, 3), (1, 2), (0, 1)]

        for i, (nq, nr) in enumerate(neighbors):
            # If neighbor is valid node
            if (nq, nr) in all_nodes:
                # If no path, draw wall
                if not maze_graph.has_edge((q, r), (nq, nr)):
                    # Get vertex indices
                    v_idx1, v_idx2 = edge_pairs[i]

                    # Calc points
                    a1 = math.radians(vertex_angles[v_idx1])
                    a2 = math.radians(vertex_angles[v_idx2])

                    x1 = cx + HEX_SIZE * math.cos(a1)
                    y1 = cy + HEX_SIZE * math.sin(a1)
                    x2 = cx + HEX_SIZE * math.cos(a2)
                    y2 = cy + HEX_SIZE * math.sin(a2)

                    svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="wall" />')
            else:
                # Boundary wall (no neighbor)
                v_idx1, v_idx2 = edge_pairs[i]
                a1 = math.radians(vertex_angles[v_idx1])
                a2 = math.radians(vertex_angles[v_idx2])
                x1 = cx + HEX_SIZE * math.cos(a1)
                y1 = cy + HEX_SIZE * math.sin(a1)
                x2 = cx + HEX_SIZE * math.cos(a2)
                y2 = cy + HEX_SIZE * math.sin(a2)
                svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="wall" />')

    # Draw Player
    if player_pos:
        px, py = hex_to_pixel(*player_pos)
        svg.append(f'<circle cx="{px}" cy="{py}" r="{HEX_SIZE * 0.6}" class="player" />')

    svg.append('</svg>')
    return "".join(svg)

# --- GAME WRAPPER ---

def app():
    if 'hex_graph' not in st.session_state: st.session_state['hex_graph'] = None
    if 'hex_pos' not in st.session_state: st.session_state['hex_pos'] = None
    if 'hex_seed' not in st.session_state: st.session_state['hex_seed'] = 42
    if 'hex_game_over' not in st.session_state: st.session_state['hex_game_over'] = False
    if 'hex_starts' not in st.session_state: st.session_state['hex_starts'] = []
    if 'hex_ends' not in st.session_state: st.session_state['hex_ends'] = []
    if 'hex_solution' not in st.session_state: st.session_state['hex_solution'] = None

    st.title("🐝 Hexagonal Maze")
    st.markdown("Navigate from any **Green** starting cell to any **Red** exit cell.")

    with st.sidebar:
        st.header("Maze Settings")
        st.number_input("Seed", 0, 99999, key='hex_seed')
        radius = st.slider("Radius", 2, 8, 4)

        c1, c2 = st.columns(2)
        n_starts = c1.number_input("Entrances", 1, 6, 2)
        n_ends = c2.number_input("Exits", 1, 6, 2)

        if st.button("Generate Maze", type="primary"):
            G, starts, ends = HexMazeGenerator.generate(radius, st.session_state['hex_seed'], n_starts, n_ends)
            st.session_state['hex_graph'] = G
            st.session_state['hex_starts'] = starts
            st.session_state['hex_ends'] = ends
            # Spawn at first start
            st.session_state['hex_pos'] = starts[0]
            st.session_state['hex_game_over'] = False
            st.session_state['hex_solution'] = None

    if st.session_state['hex_graph']:
        col_game, col_ctrl = st.columns([2, 1])

        with col_game:
            svg_html = generate_svg(
                st.session_state['hex_graph'],
                st.session_state['hex_pos'],
                st.session_state['hex_starts'],
                st.session_state['hex_ends'],
                st.session_state['hex_solution']
            )
            st.markdown(svg_html, unsafe_allow_html=True)

        with col_ctrl:
            st.subheader("Controls")
            st.write(f"Current Position: {st.session_state['hex_pos']}")

            # Hex Movement is 6-way.
            # Grid Layout for buttons:
            #   NW  NE
            # W       E
            #   SW  SE

            def move_hex(dq, dr):
                if st.session_state['hex_game_over']: return
                curr_q, curr_r = st.session_state['hex_pos']
                next_node = (curr_q + dq, curr_r + dr)

                # Check path exists in graph (no wall)
                G = st.session_state['hex_graph']
                if G.has_edge((curr_q, curr_r), next_node):
                    st.session_state['hex_pos'] = next_node

                    if next_node in st.session_state['hex_ends']:
                        st.session_state['hex_game_over'] = True
                        st.balloons()
                        st.toast("Escaped the Maze!")
                else:
                    st.toast("Blocked!", icon="🚫")

            # Top Row
            c1, c2 = st.columns(2)
            c1.button("↖ NW", use_container_width=True, on_click=move_hex, args=(0, -1)) # Wait, 0,-1 is Top Left?
            # Directions: (+1, 0)R, (+1, -1)TR, (0, -1)TL, (-1, 0)L, (-1, +1)BL, (0, +1)BR
            # NW is (0, -1)? No, (0,-1) is TopLeft. (-1, 0) is Left.
            # Visual check:
            # q=0,r=0.
            # q=0,r=-1 -> x=0, y=-sqrt3. UP. Actually North.
            # Let's use simple labels based on Neighbor logic
            # (0,-1) -> Top Left (NW)
            # (+1,-1) -> Top Right (NE)
            # (-1,0) -> Left (W)
            # (+1,0) -> Right (E)
            # (-1,+1) -> Bot Left (SW)
            # (0,+1) -> Bot Right (SE)

            c2.button("↗ NE", use_container_width=True, on_click=move_hex, args=(1, -1))

            # Mid Row
            c3, c4 = st.columns(2)
            c3.button("⬅ W", use_container_width=True, on_click=move_hex, args=(-1, 0))
            c4.button("➡ E", use_container_width=True, on_click=move_hex, args=(1, 0))

            # Bot Row
            c5, c6 = st.columns(2)
            c5.button("↙ SW", use_container_width=True, on_click=move_hex, args=(-1, 1))
            c6.button("↘ SE", use_container_width=True, on_click=move_hex, args=(0, 1))

            st.divider()
            if st.button("🏳️ Give Up (Solve)"):
                # Find path from current pos to ANY end
                G = st.session_state['hex_graph']
                curr = st.session_state['hex_pos']

                shortest = None
                for end_node in st.session_state['hex_ends']:
                    try:
                        path = nx.shortest_path(G, curr, end_node)
                        if shortest is None or len(path) < len(shortest):
                            shortest = path
                    except: pass

                if shortest:
                    st.session_state['hex_solution'] = shortest
                    st.rerun()
                else:
                    st.error("No path found!")

            if st.session_state['hex_game_over']:
                st.success("Maze Solved!")

    else:
        st.info("👈 Generate a maze to start!")
