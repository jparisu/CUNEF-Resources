import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from collections import deque
import streamlit.components.v1 as components

# --- CONFIGURATION & CONSTANTS ---
TILE_EMPTY = 0
TILE_WALL = 1
TILE_MUD = 2
TILE_START = 3
TILE_DEST = 4

COLORS = {
    TILE_EMPTY: "#FFFFFF",  # White
    TILE_WALL: "#2C3E50",   # Dark Blue/Grey
    TILE_MUD: "#D35400",    # Burnt Orange
    TILE_START: "#27AE60",  # Green
    TILE_DEST: "#C0392B",   # Red
}

DIRECTIONS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# --- LOGIC CLASSES ---

class GameEngine:
    @staticmethod
    def get_slide_result(board, start_pos, direction):
        """
        Returns (end_pos, distance_traveled)
        Slide until Wall, Border, or stop ON Mud/Start/Dest.
        """
        rows, cols = board.shape
        dr, dc = DIRECTIONS[direction]
        r, c = start_pos
        dist = 0

        while True:
            nr, nc = r + dr, c + dc

            # Check boundaries
            if not (0 <= nr < rows and 0 <= nc < cols):
                break # Stop at edge

            # Check Wall
            if board[nr, nc] == TILE_WALL:
                break # Stop before wall

            # Move
            r, c = nr, nc
            dist += 1

            # Check Mud/Start/Dest (They act as "Sticky" tiles)
            cell_type = board[r, c]
            if cell_type in [TILE_MUD, TILE_START, TILE_DEST]:
                break

        return (r, c), dist

class Solver:
    def __init__(self, board):
        self.board = board
        self.rows, self.cols = board.shape

    def build_graph(self, start_node):
        """
        Builds a directed graph using BFS starting from start_node.
        Nodes are (r, c) tuples.
        """
        G = nx.DiGraph()
        queue = deque([start_node])
        visited = {start_node}

        G.add_node(start_node)

        while queue:
            curr = queue.popleft()

            for direction in DIRECTIONS:
                next_pos, dist = GameEngine.get_slide_result(self.board, curr, direction)

                if dist > 0: # Only add valid moves
                    # Add edge (we add edge even if visited to show connectivity)
                    G.add_edge(curr, next_pos, weight=dist, label=direction)

                    if next_pos not in visited:
                        visited.add(next_pos)
                        queue.append(next_pos)

        return G

    def dijkstra(self, start, end):
        """
        Finds shortest path based on tiles traversed (weight).
        Returns (list_of_moves_with_targets, total_tiles).
        """
        G = self.build_graph(start)

        try:
            path = nx.dijkstra_path(G, start, end, weight='weight')
            length = nx.dijkstra_path_length(G, start, end, weight='weight')

            # Convert node path to list of moves (directions)
            moves = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Get edge data
                edge_data = G[u][v]
                moves.append((edge_data['label'], v))

            return moves, length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, 0

    def get_path_nodes(self, start, end):
        """Helper to get the actual list of nodes in the optimal path."""
        G = self.build_graph(start)
        try:
            return nx.dijkstra_path(G, start, end, weight='weight'), G
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, G

class LevelGenerator:
    @staticmethod
    def generate(rows, cols, seed, walls_prop, mud_prop, mud_enabled, ensure_solution,
                 start_pos_req=None, dest_pos_req=None, min_moves=0, mode="Standard"):
        random.seed(seed)
        np.random.seed(seed)

        tries = 0
        # Increase retries for branched mode as it's harder to satisfy
        max_tries = 2000 if (ensure_solution or mode == "Branched") else 1

        while tries < max_tries:
            # 1. Initialize empty board
            board = np.full((rows, cols), TILE_EMPTY)

            # 2. Determine Start/Dest
            if start_pos_req:
                s_r, s_c = start_pos_req
                if not (0 <= s_r < rows and 0 <= s_c < cols):
                     raise ValueError(f"Start position {start_pos_req} is out of bounds.")
            else:
                s_r, s_c = random.randint(0, rows-1), random.randint(0, cols-1)

            if dest_pos_req:
                d_r, d_c = dest_pos_req
                if not (0 <= d_r < rows and 0 <= d_c < cols):
                     raise ValueError(f"Destination position {dest_pos_req} is out of bounds.")
            else:
                d_r, d_c = random.randint(0, rows-1), random.randint(0, cols-1)
                # Ensure random dest is not start
                while (d_r, d_c) == (s_r, s_c):
                    d_r, d_c = random.randint(0, rows-1), random.randint(0, cols-1)

            # 3. Place Walls
            num_cells = rows * cols
            num_walls = int(num_cells * walls_prop)

            # Create list of all coordinates
            all_coords = [(r, c) for r in range(rows) for c in range(cols)]
            if (s_r, s_c) in all_coords: all_coords.remove((s_r, s_c))
            if (d_r, d_c) in all_coords: all_coords.remove((d_r, d_c))

            random.shuffle(all_coords)
            walls_coords = all_coords[:num_walls]

            for r, c in walls_coords:
                board[r, c] = TILE_WALL

            # 4. Place Mud
            if mud_enabled:
                remaining_coords = all_coords[num_walls:]
                num_mud = int(num_cells * mud_prop)
                mud_coords = remaining_coords[:num_mud]
                for r, c in mud_coords:
                    board[r, c] = TILE_MUD

            # Set Start and Dest (Logic types)
            board[s_r, s_c] = TILE_START
            board[d_r, d_c] = TILE_DEST

            # 5. Validation Logic
            if ensure_solution:
                solver = Solver(board)

                # Get Path Nodes AND Graph
                path_nodes, G = solver.get_path_nodes((s_r, s_c), (d_r, d_c))

                if path_nodes is not None:
                    # Check 1: Min Moves
                    # Path nodes includes start, so length-1 is number of moves/edges
                    if (len(path_nodes) - 1) >= min_moves:

                        # Check 2: Branched Mode Logic
                        if mode == "Branched":
                            # To ensure multiple paths, we temporarily remove a critical node
                            # from the optimal path and see if another path exists.
                            # We check the "middle" node to force a distinct divergence.
                            if len(path_nodes) > 2:
                                # Pick a node in the middle of the optimal path
                                mid_index = len(path_nodes) // 2
                                block_node = path_nodes[mid_index]

                                # Create a view of the graph without this node
                                G_sub = G.copy()
                                G_sub.remove_node(block_node)

                                # Check if path still exists
                                if nx.has_path(G_sub, (s_r, s_c), (d_r, d_c)):
                                    # Success: We have an optimal path, AND an alternative path exists
                                    return board, (s_r, s_c), (d_r, d_c)
                                # Else: It was a unique bottleneck, regenerate
                            else:
                                # Path too short to branch effectively, but technically valid if user asked for short path
                                # We'll allow it if min_moves was low, otherwise regenerate
                                if min_moves <= 1:
                                     return board, (s_r, s_c), (d_r, d_c)
                        else:
                            # Standard Mode: Just needs a solution
                            return board, (s_r, s_c), (d_r, d_c)

            else:
                return board, (s_r, s_c), (d_r, d_c)

            tries += 1

        raise Exception(f"Could not generate a map with mode '{mode}' and >={min_moves} moves in {max_tries} tries. Try lowering wall proportion.")

# --- UI HELPER FUNCTIONS ---

@st.dialog("Game Instructions")
def show_help_dialog():
    st.markdown("""
    ### 🎮 How to Play

    **Goal:** Slide your character (☺) from the **Start (S)** to the **Destination (D)**.

    **Mechanics:**
    * **Sliding:** When you choose a direction, you slide **all the way** until you hit a wall or the map edge.
    * **Stops:** * **Walls (Dark Grey):** Block movement completely.
        * **Mud (Orange ≈):** 'Sticky' tiles. You slide *through* empty space but stop *on* mud, start, or destination tiles.

    **Controls:**
    * Use the on-screen buttons.
    * Use **Arrow Keys** or **WASD** on your keyboard.

    **Generation Modes:**
    * **Standard:** Randomly generates a solvable map. Often results in a single valid path.
    * **Branched:** Ensures the map has **at least two distinct paths** to the destination. Finding the *optimal* (shortest) one becomes the real challenge.
    """)

def draw_board_html(board, player_pos):
    rows, cols = board.shape

    # CSS for the grid
    html = f"""
    <style>
        .grid-container {{
            display: grid;
            grid-template-columns: repeat({cols}, 40px);
            grid-gap: 2px;
            background-color: #333;
            padding: 5px;
            border-radius: 5px;
            width: fit-content;
        }}
        .grid-item {{
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: monospace;
            font-weight: bold;
            border-radius: 3px;
            font-size: 20px;
        }}
        .player {{
            border: 3px solid #F1C40F;
            box-shadow: 0 0 8px #F1C40F;
            z-index: 10;
        }}
    </style>
    <div class="grid-container">
    """

    for r in range(rows):
        for c in range(cols):
            cell_type = board[r, c]
            bg_color = COLORS.get(cell_type, "#FFFFFF")
            text_color = "white" if cell_type == TILE_WALL else "black"

            content = ""
            if cell_type == TILE_START: content = "S"
            elif cell_type == TILE_DEST: content = "D"
            elif cell_type == TILE_MUD: content = "≈"

            is_player = (r, c) == player_pos
            player_class = "player" if is_player else ""

            # If player is here, show player icon
            if is_player:
                content = "☺"

            html += f'<div class="grid-item {player_class}" style="background-color: {bg_color}; color: {text_color}">{content}</div>'

    html += "</div>"
    return html

def draw_graph(G, current_pos, path_history):
    """
    Draws the logic graph using matplotlib.
    """
    if G is None or len(G.nodes) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    # Layout - we try to respect the grid position slightly but allow flex
    # Flip Y so row 0 is at top
    pos = {node: (node[1], -node[0]) for node in G.nodes()}

    # Default node colors
    node_colors = ['lightgray' for _ in G.nodes()]
    node_sizes = [300 for _ in G.nodes()]

    # Draw basic graph
    # connectionstyle='arc3,rad=0.1' curves edges to avoid overlap on bidirectional paths
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, width=1, ax=ax, arrowsize=15, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # --- Highlight Path ---
    # Extract nodes from history dicts
    # history item structure: {'pos': (r,c), 'move': 'UP', 'dist': 2}
    history_nodes = [item['pos'] for item in path_history]

    if len(history_nodes) > 1:
        # Create edge list from history
        path_edges = []
        for i in range(len(history_nodes) - 1):
            u = history_nodes[i]
            v = history_nodes[i+1]
            if G.has_edge(u, v):
                path_edges.append((u, v))

        nx.draw_networkx_nodes(G, pos, nodelist=history_nodes, node_color='#a8d5e2', node_size=300, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='#2980b9', width=2.5, ax=ax, connectionstyle="arc3,rad=0.1")

    # Highlight current position
    nx.draw_networkx_nodes(G, pos, nodelist=[current_pos], node_color='#F1C40F', node_size=450, ax=ax)

    # Edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    ax.axis('off')
    return fig

# --- KEYBOARD JS INJECTION ---
def inject_keyboard_listener():
    # This JS listens for keys and finds the corresponding button by its text content and clicks it.
    js = """
    <script>
    const doc = window.parent.document;

    if (!doc.gameListenerAttached) {
        doc.addEventListener('keydown', function(e) {
            const keyMap = {
                'ArrowUp': '⬆️ Up',
                'w': '⬆️ Up',
                'W': '⬆️ Up',
                'ArrowDown': '⬇️ Down',
                's': '⬇️ Down',
                'S': '⬇️ Down',
                'ArrowLeft': '⬅️ Left',
                'a': '⬅️ Left',
                'A': '⬅️ Left',
                'ArrowRight': '➡️ Right',
                'd': '➡️ Right',
                'D': '➡️ Right'
            };

            if (keyMap[e.key]) {
                const buttons = Array.from(doc.querySelectorAll('button'));
                const targetBtn = buttons.find(btn => btn.innerText.includes(keyMap[e.key]));
                if (targetBtn) {
                    targetBtn.click();
                }
            }
        });
        doc.gameListenerAttached = true;
    }
    </script>
    """
    components.html(js, height=0, width=0)

# --- MAIN APP LOGIC ---

def init_session():
    if 'board' not in st.session_state:
        st.session_state['board'] = None
    if 'player_pos' not in st.session_state:
        st.session_state['player_pos'] = None
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'moves_count' not in st.session_state:
        st.session_state['moves_count'] = 0
    if 'tiles_count' not in st.session_state:
        st.session_state['tiles_count'] = 0
    if 'game_over' not in st.session_state:
        st.session_state['game_over'] = False
    if 'graph_obj' not in st.session_state:
        st.session_state['graph_obj'] = None
    if 'seed' not in st.session_state:
        st.session_state['seed'] = 0
    if 'level_rows' not in st.session_state:
        st.session_state['level_rows'] = 6
    if 'level_cols' not in st.session_state:
        st.session_state['level_cols'] = 6

def move_player_callback(direction):
    """
    Callback function to handle movement BEFORE rendering.
    """
    if st.session_state['game_over'] or st.session_state['board'] is None:
        return

    board = st.session_state['board']
    curr = st.session_state['player_pos']

    new_pos, dist = GameEngine.get_slide_result(board, curr, direction)

    if dist > 0:
        st.session_state['player_pos'] = new_pos
        st.session_state['moves_count'] += 1
        st.session_state['tiles_count'] += dist
        st.session_state['history'].append({'pos': new_pos, 'move': direction, 'dist': dist})

        if new_pos == st.session_state['dest_pos']:
            st.session_state['game_over'] = True
            st.balloons()
            st.toast("🎉 Destination Reached!", icon='🏆')

def reset_game():
    if st.session_state['board'] is not None:
        st.session_state['player_pos'] = st.session_state['start_pos']
        st.session_state['history'] = [{'pos': st.session_state['start_pos'], 'move': 'Start', 'dist': 0}]
        st.session_state['moves_count'] = 0
        st.session_state['tiles_count'] = 0
        st.session_state['game_over'] = False

def generate_new_level(rows, cols, seed, walls_prop, mud_prop, mud_enabled, ensure_sol,
                       start_req, dest_req, min_moves, gen_mode):
    try:
        # Parse inputs if strings
        s_pos = None
        if start_req:
            try:
                parts = [int(x.strip()) for x in start_req.split(',')]
                if len(parts) == 2: s_pos = tuple(parts)
            except: pass

        d_pos = None
        if dest_req:
            try:
                parts = [int(x.strip()) for x in dest_req.split(',')]
                if len(parts) == 2: d_pos = tuple(parts)
            except: pass

        b, s, d = LevelGenerator.generate(
            rows, cols, seed, walls_prop, mud_prop, mud_enabled, ensure_sol,
            start_pos_req=s_pos, dest_pos_req=d_pos, min_moves=min_moves, mode=gen_mode
        )
        st.session_state['board'] = b
        st.session_state['start_pos'] = s
        st.session_state['dest_pos'] = d
        st.session_state['player_pos'] = s
        st.session_state['history'] = [{'pos': s, 'move': 'Start', 'dist': 0}]
        st.session_state['moves_count'] = 0
        st.session_state['tiles_count'] = 0
        st.session_state['game_over'] = False

        # Precompute Graph
        solver = Solver(b)
        st.session_state['graph_obj'] = solver.build_graph(s)
    except Exception as e:
        st.error(str(e))

def update_size_callback():
    # Syncs the simple size slider to rows/cols
    size = st.session_state['simple_size']
    st.session_state['level_rows'] = size
    st.session_state['level_cols'] = size

def randomize_seed_callback():
    st.session_state['seed'] = random.randint(0, 999999)

def main():
    st.set_page_config(page_title="Sliding Puzzle", layout="wide")
    init_session()

    # Inject Keyboard Listeners
    inject_keyboard_listener()

    # --- HEADER ---
    col_title, col_help = st.columns([10, 1])
    with col_title:
        st.title("🧊 Sliding Puzzle Game")
    with col_help:
        if st.button("❓", help="How to play"):
            show_help_dialog()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Settings")

        # Standard Settings
        st.number_input("Seed", 0, 999999, key='seed')
        if st.button("🎲 Randomize Seed", use_container_width=True, on_click=randomize_seed_callback):
            pass # Callback handles logic

        gen_mode = st.radio("Generation Mode", ["Standard", "Branched"], help="Branched ensures multiple paths exist.")

        st.slider("Map Size", 3, 15, 6, key='simple_size', on_change=update_size_callback)
        mud_enabled = st.checkbox("Enable Mud Tiles", value=False)

        st.markdown("---")

        # Advanced Settings
        with st.expander("🛠️ Advanced Parameters", expanded=False):
            st.caption("Map Dimensions")
            c_r, c_c = st.columns(2)
            rows = c_r.number_input("Rows", 3, 20, key='level_rows')
            cols = c_c.number_input("Cols", 3, 20, key='level_cols')

            st.caption("Positions (Format: row,col)")
            start_txt = st.text_input("Start Position", placeholder="Random (e.g. 0,0)")
            dest_txt = st.text_input("Dest Position", placeholder="Random (e.g. 5,5)")

            st.caption("Difficulty")
            walls_prop = st.slider("Walls %", 0.0, 0.5, 0.2)
            mud_prop = st.slider("Mud %", 0.0, 0.5, 0.1) if mud_enabled else 0.0
            min_moves = st.number_input("Min Solution Moves", 1, 20, 3, help="Regenerates if solution is too short.")
            ensure_sol = st.checkbox("Ensure Solvable", value=True)

        st.markdown("---")

        col_gen, col_reset = st.columns(2)
        if col_gen.button("Generate Level", type="primary"):
            generate_new_level(rows, cols, st.session_state['seed'], walls_prop, mud_prop, mud_enabled, ensure_sol, start_txt, dest_txt, min_moves, gen_mode)

        if col_reset.button("Reset Game"):
            reset_game()

    # --- MAIN INTERFACE ---

    if st.session_state['board'] is not None:

        col_board, col_info = st.columns([1, 1])

        # Board Container Placeholder for Animation
        board_placeholder = col_board.empty()

        # 1. RENDER BOARD
        with board_placeholder.container():
            st.subheader("Game Board")
            st.markdown(draw_board_html(st.session_state['board'], st.session_state['player_pos']), unsafe_allow_html=True)

            st.write("")
            st.write("**Controls:**")

            # Button Grid
            c1, c2, c3 = st.columns([1, 1, 1])
            with c2:
                st.button("⬆️ Up", use_container_width=True, key="btn_up", on_click=move_player_callback, args=('UP',))

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.button("⬅️ Left", use_container_width=True, key="btn_left", on_click=move_player_callback, args=('LEFT',))
            with c2:
                st.button("⬇️ Down", use_container_width=True, key="btn_down", on_click=move_player_callback, args=('DOWN',))
            with c3:
                st.button("➡️ Right", use_container_width=True, key="btn_right", on_click=move_player_callback, args=('RIGHT',))

        with col_info:
            st.subheader("Game Status")

            if st.session_state['game_over']:
                st.success(f"🏆 **Destination Reached!**\n\n"
                           f"- Total Moves: {st.session_state['moves_count']}\n"
                           f"- Tiles Slid: {st.session_state['tiles_count']}")
            else:
                st.info(f"📍 **Position:** {st.session_state['player_pos']}\n\n"
                        f"👣 **Moves:** {st.session_state['moves_count']} | "
                        f"📏 **Tiles:** {st.session_state['tiles_count']}")

            st.caption("Path History")
            hist_moves = [h['move'] for h in st.session_state['history'] if h['move'] != 'Start']
            st.text_area("Log", str(hist_moves), height=100, disabled=True)

            st.divider()

            # Optimal Solution Viewer
            if st.button("▶️ Show Optimal Solution Animation"):
                solver = Solver(st.session_state['board'])
                moves_list, _ = solver.dijkstra(st.session_state['start_pos'], st.session_state['dest_pos'])

                if moves_list:
                    # Reset game state logic immediately for the animation
                    reset_game()

                    # Animate using placeholder
                    for move_dir, target in moves_list:
                        # Logic step
                        new_pos, dist = GameEngine.get_slide_result(st.session_state['board'], st.session_state['player_pos'], move_dir)

                        # Update stats
                        st.session_state['player_pos'] = new_pos
                        st.session_state['moves_count'] += 1
                        st.session_state['tiles_count'] += dist
                        st.session_state['history'].append({'pos': new_pos, 'move': move_dir, 'dist': dist})

                        # Render step
                        with board_placeholder.container():
                            st.subheader("Game Board")
                            st.markdown(draw_board_html(st.session_state['board'], new_pos), unsafe_allow_html=True)
                            st.write(f"🤖 AI Moving: **{move_dir}**...")

                        time.sleep(0.4)

                    # Apply final state to session
                    st.session_state['game_over'] = True
                    st.balloons()
                    st.rerun()
                else:
                    st.warning("No solution found!")

        # --- GRAPH SECTION ---
        st.write("")
        with st.expander("📊 Logic Graph Visualization", expanded=False):
            if st.session_state['graph_obj']:
                fig = draw_graph(st.session_state['graph_obj'],
                                 st.session_state['player_pos'],
                                 st.session_state['history'])
                if fig:
                    st.pyplot(fig)
            else:
                st.write("Graph not generated yet.")

    else:
        st.info("👈 Please Click 'Generate Level' in the sidebar to start!")

if __name__ == "__main__":
    main()
