# 🧊 Sliding Puzzle Game

A procedural puzzle game built with Python and Streamlit. The player must navigate a grid by sliding in cardinal directions. The catch? You cannot stop until you hit a wall or the edge of the map.

## 🚀 Getting Started

### Prerequisites

You need Python installed on your system along with the following libraries:

``` bash
pip install streamlit numpy networkx matplotlib
```

### Running the Game

To launch the game interface, run the following command in your terminal:

``` bash
streamlit run sliding_map.py
```

The game will open automatically in your default web browser (usually at <http://localhost:8501>).


## 🎮 User Guide

### Objective

Navigate your character (☺) from the Start point (S) to the Destination (D).

### Mechanics

- Frictionless Movement: When you choose a direction (Up, Down, Left, Right), you slide all the way until you are stopped by:
    - A Wall (Dark Grey Block)
    - The edge of the map
    - A "Sticky" Tile (Mud, Start, or Destination)

- Mud Tiles (≈): These are sticky spots. You can pass through them if they are in the middle of your slide, but if your slide ends exactly on one, you stop there.


### Controls

- UI Buttons: Click the arrow buttons on the screen.
- Keyboard: Use Arrow Keys or WASD. (Note: The game window must be active for keyboard inputs to register).

### Generation Modes

- Standard Mode: Generates a random map that is guaranteed to be solvable. These maps often feature linear paths.
- Branched Mode: A "smarter" generation algorithm. It ensures there are at least two distinct paths to the destination. This forces you to make strategic choices between a shorter optimal path and longer valid ones.

### 🛠️ Developer Design

The project is contained within a single script (sliding_game.py) designed to be stateless in logic but stateful in session management via Streamlit.


1. Architecture

The application is split into three main logic classes and a UI layer.

- `GameEngine`: Handles the physics of the game.
    - get_slide_result(board, start_pos, direction): Calculates the end coordinate of a move by iterating through the grid until a blocking condition is met.

- `Solver`: Graph theory implementation.
    - Treats the grid as a Directed Graph where Nodes are Grid Coordinates and Edges are Valid Slides.
    - Uses BFS to build the graph of reachable states.
    - Uses Dijkstra's Algorithm to find the path with the minimum "weight" (tiles traveled).

- `LevelGenerator`: Procedural Generation.
    - Uses Rejection Sampling: It generates a random board and validates it against criteria (solvability, min moves).
    - Branched Logic: It solves the map, identifies the optimal path, and then verifies that removing a node from that path leaves at least one other valid path.


2. State Management

To maintain the game state across Streamlit's reruns, we use `st.session_state`:

- `board`: The Numpy array representing the grid.
- `player_pos`: Tuple (row, col) of current position.
- `history`: List of move metadata (for the log and game status tracking).
- `graph_obj`: Cached NetworkX graph object.


3. Visualization

- Grid: Rendered using raw HTML/CSS injected via `st.markdown` for a tight, responsive layout.
- Graph: Visualized using `matplotlib` and `networkx`, mapping logical nodes to their physical grid coordinates.


4. Animation

The "Optimal Solution" feature uses an `st.empty()` container to update frames in a loop with `time.sleep()`, providing a smooth visual walkthrough of the AI's path.
