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
        rows, cols = board.shape
        dr, dc = DIRECTIONS[direction]
        r, c = start_pos
        dist = 0

        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols): break
            if board[nr, nc] == TILE_WALL: break
            r, c = nr, nc
            dist += 1
            cell_type = board[r, c]
            if cell_type in [TILE_MUD, TILE_START, TILE_DEST]: break

        return (r, c), dist

class Solver:
    def __init__(self, board):
        self.board = board
        self.rows, self.cols = board.shape

    def build_graph(self, start_node):
        G = nx.DiGraph()
        queue = deque([start_node])
        visited = {start_node}
        G.add_node(start_node)

        while queue:
            curr = queue.popleft()
            for direction in DIRECTIONS:
                next_pos, dist = GameEngine.get_slide_result(self.board, curr, direction)
                if dist > 0:
                    G.add_edge(curr, next_pos, weight=dist, label=direction)
                    if next_pos not in visited:
                        visited.add(next_pos)
                        queue.append(next_pos)
        return G

    def dijkstra(self, start, end):
        G = self.build_graph(start)
        try:
            path = nx.dijkstra_path(G, start, end, weight='weight')
            length = nx.dijkstra_path_length(G, start, end, weight='weight')
            moves = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = G[u][v]
                moves.append((edge_data['label'], v))
            return moves, length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, 0

    def get_path_nodes(self, start, end):
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
        max_tries = 2000 if (ensure_solution or mode == "Branched") else 1

        while tries < max_tries:
            board = np.full((rows, cols), TILE_EMPTY)

            if start_pos_req:
                s_r, s_c = start_pos_req
                if not (0 <= s_r < rows and 0 <= s_c < cols): raise ValueError("Start out of bounds")
            else:
                s_r, s_c = random.randint(0, rows-1), random.randint(0, cols-1)

            if dest_pos_req:
                d_r, d_c = dest_pos_req
                if not (0 <= d_r < rows and 0 <= d_c < cols): raise ValueError("Dest out of bounds")
            else:
                d_r, d_c = random.randint(0, rows-1), random.randint(0, cols-1)
                while (d_r, d_c) == (s_r, s_c):
                    d_r, d_c = random.randint(0, rows-1), random.randint(0, cols-1)

            num_cells = rows * cols
            num_walls = int(num_cells * walls_prop)
            all_coords = [(r, c) for r in range(rows) for c in range(cols)]
            if (s_r, s_c) in all_coords: all_coords.remove((s_r, s_c))
            if (d_r, d_c) in all_coords: all_coords.remove((d_r, d_c))

            random.shuffle(all_coords)
            walls_coords = all_coords[:num_walls]
            for r, c in walls_coords: board[r, c] = TILE_WALL

            if mud_enabled:
                remaining_coords = all_coords[num_walls:]
                num_mud = int(num_cells * mud_prop)
                mud_coords = remaining_coords[:num_mud]
                for r, c in mud_coords: board[r, c] = TILE_MUD

            board[s_r, s_c] = TILE_START
            board[d_r, d_c] = TILE_DEST

            if ensure_solution:
                solver = Solver(board)
                path_nodes, G = solver.get_path_nodes((s_r, s_c), (d_r, d_c))
                if path_nodes is not None:
                    if (len(path_nodes) - 1) >= min_moves:
                        if mode == "Branched":
                            if len(path_nodes) > 2:
                                mid_index = len(path_nodes) // 2
                                block_node = path_nodes[mid_index]
                                G_sub = G.copy()
                                G_sub.remove_node(block_node)
                                if nx.has_path(G_sub, (s_r, s_c), (d_r, d_c)):
                                    return board, (s_r, s_c), (d_r, d_c)
                            else:
                                if min_moves <= 1: return board, (s_r, s_c), (d_r, d_c)
                        else:
                            return board, (s_r, s_c), (d_r, d_c)
            else:
                return board, (s_r, s_c), (d_r, d_c)
            tries += 1

        raise Exception(f"Could not generate map (Mode: {mode}, Tries: {max_tries}).")

# --- UI HELPER FUNCTIONS ---

def draw_board_html(board, player_pos):
    rows, cols = board.shape
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
            if is_player: content = "☺"
            html += f'<div class="grid-item {player_class}" style="background-color: {bg_color}; color: {text_color}">{content}</div>'
    html += "</div>"
    return html

def draw_graph(G, current_pos, path_history):
    if G is None or len(G.nodes) == 0: return None
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = {node: (node[1], -node[0]) for node in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightgray', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, width=1, ax=ax, arrowsize=15, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    history_nodes = [item['pos'] for item in path_history]
    if len(history_nodes) > 1:
        path_edges = []
        for i in range(len(history_nodes) - 1):
            u, v = history_nodes[i], history_nodes[i+1]
            if G.has_edge(u, v): path_edges.append((u, v))
        nx.draw_networkx_nodes(G, pos, nodelist=history_nodes, node_color='#a8d5e2', node_size=300, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='#2980b9', width=2.5, ax=ax, connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_nodes(G, pos, nodelist=[current_pos], node_color='#F1C40F', node_size=450, ax=ax)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.axis('off')
    return fig

def inject_keyboard_listener():
    js = """
    <script>
    const doc = window.parent.document;
    if (!doc.gameListenerAttached) {
        doc.addEventListener('keydown', function(e) {
            const keyMap = {'ArrowUp': '⬆️ Up', 'w': '⬆️ Up', 'ArrowDown': '⬇️ Down', 's': '⬇️ Down',
                            'ArrowLeft': '⬅️ Left', 'a': '⬅️ Left', 'ArrowRight': '➡️ Right', 'd': '➡️ Right'};
            if (keyMap[e.key]) {
                const buttons = Array.from(doc.querySelectorAll('button'));
                const targetBtn = buttons.find(btn => btn.innerText.includes(keyMap[e.key]));
                if (targetBtn) targetBtn.click();
            }
        });
        doc.gameListenerAttached = true;
    }
    </script>
    """
    components.html(js, height=0, width=0)

# --- APP WRAPPER ---
def app():
    if 'slide_board' not in st.session_state: st.session_state['slide_board'] = None
    if 'slide_player_pos' not in st.session_state: st.session_state['slide_player_pos'] = None
    if 'slide_history' not in st.session_state: st.session_state['slide_history'] = []
    if 'slide_moves_count' not in st.session_state: st.session_state['slide_moves_count'] = 0
    if 'slide_tiles_count' not in st.session_state: st.session_state['slide_tiles_count'] = 0
    if 'slide_game_over' not in st.session_state: st.session_state['slide_game_over'] = False
    if 'slide_graph_obj' not in st.session_state: st.session_state['slide_graph_obj'] = None
    if 'slide_seed' not in st.session_state: st.session_state['slide_seed'] = 0

    inject_keyboard_listener()

    st.title("🧊 Sliding Puzzle Game")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Settings")
        st.number_input("Seed", 0, 999999, key='slide_seed')
        if st.button("🎲 Randomize Seed", use_container_width=True):
            st.session_state['slide_seed'] = random.randint(0, 999999)
            st.rerun()

        gen_mode = st.radio("Generation Mode", ["Standard", "Branched"])
        size = st.slider("Map Size", 3, 15, 6)
        mud_enabled = st.checkbox("Enable Mud Tiles", value=False)

        with st.expander("🛠️ Advanced Parameters", expanded=False):
            c_r, c_c = st.columns(2)
            rows = c_r.number_input("Rows", 3, 20, value=size)
            cols = c_c.number_input("Cols", 3, 20, value=size)
            start_txt = st.text_input("Start Position", placeholder="Random (e.g. 0,0)")
            dest_txt = st.text_input("Dest Position", placeholder="Random (e.g. 5,5)")
            walls_prop = st.slider("Walls %", 0.0, 0.5, 0.2)
            mud_prop = st.slider("Mud %", 0.0, 0.5, 0.1) if mud_enabled else 0.0
            min_moves = st.number_input("Min Solution Moves", 1, 20, 3)
            ensure_sol = st.checkbox("Ensure Solvable", value=True)

        col_gen, col_reset = st.columns(2)
        if col_gen.button("Generate Level", type="primary"):
            try:
                s_pos = tuple(map(int, start_txt.split(','))) if start_txt else None
                d_pos = tuple(map(int, dest_txt.split(','))) if dest_txt else None
            except: s_pos, d_pos = None, None

            try:
                b, s, d = LevelGenerator.generate(rows, cols, st.session_state['slide_seed'], walls_prop, mud_prop,
                                                mud_enabled, ensure_sol, s_pos, d_pos, min_moves, gen_mode)
                st.session_state['slide_board'] = b
                st.session_state['slide_start_pos'] = s
                st.session_state['slide_dest_pos'] = d
                st.session_state['slide_player_pos'] = s
                st.session_state['slide_history'] = [{'pos': s, 'move': 'Start', 'dist': 0}]
                st.session_state['slide_moves_count'] = 0
                st.session_state['slide_tiles_count'] = 0
                st.session_state['slide_game_over'] = False
                solver = Solver(b)
                st.session_state['slide_graph_obj'] = solver.build_graph(s)
            except Exception as e: st.error(str(e))

        if col_reset.button("Reset Game"):
            if st.session_state['slide_board'] is not None:
                st.session_state['slide_player_pos'] = st.session_state['slide_start_pos']
                st.session_state['slide_history'] = [{'pos': st.session_state['slide_start_pos'], 'move': 'Start', 'dist': 0}]
                st.session_state['slide_moves_count'] = 0
                st.session_state['slide_tiles_count'] = 0
                st.session_state['slide_game_over'] = False

    # --- MAIN INTERFACE ---
    if st.session_state['slide_board'] is not None:
        col_board, col_info = st.columns([1, 1])
        board_placeholder = col_board.empty()

        def move_cb(direction):
            if st.session_state['slide_game_over']: return
            b = st.session_state['slide_board']
            c = st.session_state['slide_player_pos']
            n, d = GameEngine.get_slide_result(b, c, direction)
            if d > 0:
                st.session_state['slide_player_pos'] = n
                st.session_state['slide_moves_count'] += 1
                st.session_state['slide_tiles_count'] += d
                st.session_state['slide_history'].append({'pos': n, 'move': direction, 'dist': d})
                if n == st.session_state['slide_dest_pos']:
                    st.session_state['slide_game_over'] = True
                    st.balloons()
                    st.toast("Destination Reached!")

        with board_placeholder.container():
            st.subheader("Game Board")
            st.markdown(draw_board_html(st.session_state['slide_board'], st.session_state['slide_player_pos']), unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 1])
            with c2: st.button("⬆️ Up", use_container_width=True, key="s_up", on_click=move_cb, args=('UP',))
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1: st.button("⬅️ Left", use_container_width=True, key="s_left", on_click=move_cb, args=('LEFT',))
            with c2: st.button("⬇️ Down", use_container_width=True, key="s_down", on_click=move_cb, args=('DOWN',))
            with c3: st.button("➡️ Right", use_container_width=True, key="s_right", on_click=move_cb, args=('RIGHT',))

        with col_info:
            st.subheader("Game Status")
            if st.session_state['slide_game_over']:
                st.success(f"🏆 Destination Reached!\nMoves: {st.session_state['slide_moves_count']} | Tiles: {st.session_state['slide_tiles_count']}")
            else:
                st.info(f"📍 Position: {st.session_state['slide_player_pos']}\nMoves: {st.session_state['slide_moves_count']} | Tiles: {st.session_state['slide_tiles_count']}")

            hist_moves = [h['move'] for h in st.session_state['slide_history'] if h['move'] != 'Start']
            st.text_area("Log", str(hist_moves), height=100, disabled=True)
            st.divider()

            if st.button("▶️ Solve"):
                solver = Solver(st.session_state['slide_board'])
                moves_list, _ = solver.dijkstra(st.session_state['slide_start_pos'], st.session_state['slide_dest_pos'])
                if moves_list:
                    reset_game_logic = True # We can't call reset button logic directly
                    if st.session_state['slide_board'] is not None:
                        st.session_state['slide_player_pos'] = st.session_state['slide_start_pos']
                        st.session_state['slide_history'] = [{'pos': st.session_state['slide_start_pos'], 'move': 'Start', 'dist': 0}]
                        st.session_state['slide_moves_count'] = 0
                        st.session_state['slide_tiles_count'] = 0
                        st.session_state['slide_game_over'] = False

                    anim_pos = st.session_state['slide_start_pos']
                    for move_dir, target in moves_list:
                        anim_pos, d = GameEngine.get_slide_result(st.session_state['slide_board'], anim_pos, move_dir)
                        st.session_state['slide_player_pos'] = anim_pos # Update state for final sync
                        st.session_state['slide_moves_count'] += 1
                        st.session_state['slide_tiles_count'] += d
                        st.session_state['slide_history'].append({'pos': anim_pos, 'move': move_dir, 'dist': d})

                        with board_placeholder.container():
                            st.subheader("Game Board")
                            st.markdown(draw_board_html(st.session_state['slide_board'], anim_pos), unsafe_allow_html=True)
                            st.write(f"🤖 AI Moving: **{move_dir}**...")
                        time.sleep(0.4)
                    st.session_state['slide_game_over'] = True
                    st.balloons()
                    st.rerun()
                else: st.warning("No solution found!")

        with st.expander("📊 Logic Graph", expanded=False):
            if st.session_state['slide_graph_obj']:
                st.pyplot(draw_graph(st.session_state['slide_graph_obj'], st.session_state['slide_player_pos'], st.session_state['slide_history']))
    else:
        st.info("👈 Please Click 'Generate Level' in the sidebar to start!")
