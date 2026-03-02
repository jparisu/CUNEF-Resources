from pathlib import Path
import sys

import streamlit as st


# Allow running `streamlit run app.py` without installing the package.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import path_finding_game_app.games.sliding_map as sliding_map
import path_finding_game_app.games.hex_maze as hex_maze

# Set page config once at the top level
st.set_page_config(page_title="Puzzle Arcade", layout="wide", page_icon="🕹️")

# --- Game Metadata Configuration ---
# This dictionary acts as the registry for all games in the suite.
# To add a new game, just import the module and add an entry here.
GAMES = {
    "sliding_map": {
        "title": "Sliding Map",
        "icon": "🧊",
        "description": "Slide your character through obstacles. Movement continues until you hit a wall. Strategy is key!",
        "module": sliding_map
    },
    "hex_maze": {
        "title": "Hexagonal Maze",
        "icon": "🐝",
        "description": "Navigate a complex honeycomb grid. Find your way from one of the green entrances to a red exit.",
        "module": hex_maze
    }
}

# --- Session State Navigation ---
# We use 'current_view' to track whether we are on the 'Home' screen or inside a specific game.
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'Home'

def set_view(view_name):
    st.session_state['current_view'] = view_name

# --- Main App Logic ---

if st.session_state['current_view'] == 'Home':
    # === HOME MENU VIEW ===
    st.title("🕹️ Puzzle Arcade")
    st.markdown("### Select a Game")
    st.markdown("---")

    # Create a grid layout for game cards
    # We dynamically create columns based on the number of games
    cols = st.columns(len(GAMES))

    for idx, (key, game) in enumerate(GAMES.items()):
        with cols[idx]:
            # Create a nice visual card for each game
            with st.container(border=True):
                st.markdown(f"# {game['icon']}")
                st.subheader(game['title'])
                st.write(game['description'])

                # The Play button redirects to the game view
                if st.button(f"Play {game['title']}", use_container_width=True, key=f"btn_{key}"):
                    set_view(key)
                    st.rerun()

else:
    # === GAME VIEW ===
    game_key = st.session_state['current_view']

    if game_key in GAMES:
        # Navigation Bar (Back Button)
        col_nav, col_spacer = st.columns([1, 6])
        with col_nav:
            if st.button("⬅️ Back to Menu", use_container_width=True):
                set_view('Home')
                st.rerun()

        st.markdown("---")

        # Execute the selected game's main logic
        # We assume every game module has an .app() function
        GAMES[game_key]['module'].app()

    else:
        st.error(f"Game '{game_key}' not found.")
        if st.button("Return Home"):
            set_view('Home')
            st.rerun()

# --- Footer / Sidebar Info ---
with st.sidebar:
    st.markdown("---")
    if st.session_state['current_view'] != 'Home':
        current_title = GAMES[st.session_state['current_view']]['title']
        st.caption(f"Currently playing: **{current_title}**")

    st.caption("Puzzle Arcade v2.0")
