import math
import random
import itertools
import time
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objs as go

# ---------- README / Help loader ----------

def load_help_markdown(path: str = "salesman_readme.md") -> str:
    """
    Load the help/README markdown from a file.
    If the file is missing, return a short warning message.
    """
    try:
        readme_path = Path(__file__).parent / "salesman_readme.md"
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            "⚠️ **salesman_readme.md** not found.\n\n"
            "Make sure the file is in the same folder as this script."
        )

# ---------- Helpers ----------

def generate_random_points(n: int) -> List[Tuple[float, float]]:
    """Generate N random (x, y) points in [0, 10]."""
    return [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(n)]


def build_label_mapping(n: int) -> Dict[str, int]:
    """Map 'A', 'B', ... to indices 0..N-1."""
    return {chr(ord('A') + i): i for i in range(n)}


def compute_path_distance(
    path: str,
    points: List[Tuple[float, float]],
    label_to_index: Dict[str, int],
) -> float:
    """Compute Euclidean distance of the given path over the points."""
    path = path.strip().upper()
    if len(path) < 2:
        return 0.0

    indices = []
    for ch in path:
        if ch not in label_to_index:
            raise ValueError(f"Invalid letter in path: {ch}")
        indices.append(label_to_index[ch])

    total = 0.0
    for i in range(len(indices) - 1):
        x1, y1 = points[indices[i]]
        x2, y2 = points[indices[i + 1]]
        dx = x2 - x1
        dy = y2 - y1
        total += math.sqrt(dx * dx + dy * dy)
    return total


def visits_all_nodes(path: str, labels: List[str]) -> bool:
    """
    Returns True if the path (string of labels) visits all nodes at least once.
    No cycle requirement.
    """
    path_clean = path.strip().upper()
    if not path_clean:
        return False
    labels_set = set(labels)
    return labels_set.issubset(set(path_clean))


def brute_force_best_path(points: List[Tuple[float, float]], labels: List[str]) -> Tuple[str, float]:
    """
    Brute-force TSP-like search:
    - Starts at labels[0] (A)
    - Visits each other node exactly once
    - Does NOT return to the start (no cycle)
    Returns (best_path_str, best_distance).
    """
    n = len(points)
    if n <= 1:
        return "".join(labels), 0.0

    best_distance = None
    best_indices = None

    # Fix start at index 0 (label A), permute the rest
    remaining_indices = list(range(1, n))

    for perm in itertools.permutations(remaining_indices):
        indices = [0] + list(perm)  # no return to 0
        total = 0.0
        for i in range(len(indices) - 1):
            x1, y1 = points[indices[i]]
            x2, y2 = points[indices[i + 1]]
            dx = x2 - x1
            dy = y2 - y1
            total += math.sqrt(dx * dx + dy * dy)

        if best_distance is None or total < best_distance:
            best_distance = total
            best_indices = indices

    best_path_str = "".join(labels[i] for i in best_indices)
    return best_path_str, best_distance


def compute_distance_matrix(points: List[Tuple[float, float]], labels: List[str]) -> pd.DataFrame:
    """Return an N×N DataFrame with distances between every pair of nodes."""
    n = len(points)
    data = []
    for i in range(n):
        row = []
        x1, y1 = points[i]
        for j in range(n):
            x2, y2 = points[j]
            if i == j:
                row.append(0.0)
            else:
                dx = x2 - x1
                dy = y2 - y1
                row.append(math.sqrt(dx * dx + dy * dy))
        data.append(row)
    df = pd.DataFrame(data, index=labels, columns=labels)
    return df


def normalize_to_permutation(path: str, labels: List[str]) -> str:
    """
    From any path string, build a permutation of all labels:
    - Keep the order of first appearance of each valid label
    - Append any missing labels at the end in natural order.
    """
    path_clean = path.strip().upper()
    result = []
    used = set()
    label_set = set(labels)

    for ch in path_clean:
        if ch in label_set and ch not in used:
            result.append(ch)
            used.add(ch)

    for l in labels:
        if l not in used:
            result.append(l)

    return "".join(result)


def local_search_step(
    path: str,
    points: List[Tuple[float, float]],
    labels: List[str],
    label_to_index: Dict[str, int],
) -> Tuple[str, float, bool]:
    """
    Perform ONE local search step:
    - Normalize path to a permutation of labels.
    - Consider neighbors obtained by swapping TWO CONSECUTIVE letters.
    - Return the best improving neighbor (first-best), or the same path if no improvement.
    Returns: (new_path_str, new_distance, improved_flag)
    """
    # Make sure we work on a permutation
    path_perm = normalize_to_permutation(path, labels)
    base_distance = compute_path_distance(path_perm, points, label_to_index)

    best_distance = base_distance
    best_path = path_perm
    improved = False

    chars = list(path_perm)
    for i in range(len(chars) - 1):
        neighbor = chars.copy()
        neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
        neighbor_str = "".join(neighbor)
        d = compute_path_distance(neighbor_str, points, label_to_index)
        if d < best_distance - 1e-12:  # small tolerance
            best_distance = d
            best_path = neighbor_str
            improved = True
            # You can break here for first-improvement, or continue for best-improvement.
            # Let's use best-improvement:
    return best_path, best_distance, improved


# ---------- Streamlit App ----------

st.set_page_config(page_title="TSP Demo", layout="wide")

st.title("Traveling Salesman Problem Demo")

# Help / README toggle state
if "show_help" not in st.session_state:
    st.session_state.show_help = False

# Sidebar help button
with st.sidebar:
    if st.button("ℹ️ Help / README"):
        # Toggle visibility
        st.session_state.show_help = not st.session_state.show_help

# If help is active, show the README at the top of the main page
if st.session_state.show_help:
    with st.expander("ℹ️ Help / README", expanded=True):
        st.markdown(load_help_markdown())

# --- Session state init ---
if "N" not in st.session_state:
    st.session_state.N = 4  # default number of points
if "seed" not in st.session_state:
    st.session_state.seed = 0
if "points" not in st.session_state:
    random.seed(st.session_state.seed)
    st.session_state.points = generate_random_points(st.session_state.N)
if "path_input" not in st.session_state:
    st.session_state.path_input = ""

# Sidebar (left column) for parameters
st.sidebar.header("Parameters")

# N selection
N_input = st.sidebar.number_input(
    "Number of points (N)",
    min_value=2,
    max_value=26,  # up to Z
    value=st.session_state.N,
    step=1,
)

# Seed selection
seed_input = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=10_000,
    value=int(st.session_state.seed),
    step=1,
)
st.session_state.seed = int(seed_input)

# Adjust number of points if N changed
if N_input != st.session_state.N:
    old_N = st.session_state.N
    old_points = st.session_state.points

    if N_input < old_N:
        # Truncate
        st.session_state.points = old_points[: N_input]
    else:
        # Extend with seeded random new points
        random.seed(st.session_state.seed)
        extra = generate_random_points(N_input - old_N)
        st.session_state.points = old_points + extra

    st.session_state.N = int(N_input)

N = st.session_state.N
points = st.session_state.points
labels = [chr(ord("A") + i) for i in range(N)]

# Random button (seeded)
if st.sidebar.button("Random points"):
    random.seed(st.session_state.seed)
    st.session_state.points = generate_random_points(N)
    points = st.session_state.points

# Editable coordinates for each point
st.sidebar.subheader("Coordinates (in [0, 10])")
updated_points = []

for i, label in enumerate(labels):
    x_val, y_val = points[i]
    col_x, col_y = st.sidebar.columns(2)
    with col_x:
        x = st.number_input(
            f"X_{label}",
            min_value=0.0,
            max_value=10.0,
            value=float(x_val),
            key=f"x_{i}",
        )
    with col_y:
        y = st.number_input(
            f"Y_{label}",
            min_value=0.0,
            max_value=10.0,
            value=float(y_val),
            key=f"y_{i}",
        )
    updated_points.append((x, y))

# Save back updated points
st.session_state.points = updated_points
points = updated_points

# Default path: simple permutation A B C ...
default_example = "".join(labels)
if not st.session_state.path_input:
    st.session_state.path_input = default_example

# If N changed drastically, ensure path only uses existing labels
allowed_chars = set(labels)
path_clean_state = st.session_state.path_input.strip().upper()
if (not path_clean_state) or (not set(path_clean_state).issubset(allowed_chars)):
    st.session_state.path_input = default_example

label_to_index = build_label_mapping(N)

# ---------- Layout: center plot + right control column ----------

col_center, col_right = st.columns([3, 1])

# ---------- Right column: path, distance, local search, best path ----------

with col_right:
    st.subheader("Path & Distance")

    st.markdown(
        "**Available points:**<br>"
        + "<br>".join(
            f"{label}: ({x:.2f}, {y:.2f})"
            for label, (x, y) in zip(labels, points)
        ),
        unsafe_allow_html=True,
    )

    # Step time for local search
    step_time = st.number_input(
        "Local search step time (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
    )

    # Buttons: brute-force best, local search auto, local search one step
    best_button = st.button("Find best path (brute force)")
    local_auto_button = st.button("Solve with local search (auto)")
    local_step_button = st.button("Local search – one step")

    # Handle buttons BEFORE the text input so updates are visible immediately
    if best_button:
        if N > 10:
            st.warning(
                "Brute-force search is limited to N ≤ 10 to avoid huge computation. "
                "Reduce N to use this feature."
            )
        else:
            best_path_str, best_dist = brute_force_best_path(points, labels)
            st.session_state.path_input = best_path_str
            st.info(f"Brute-force best distance: {best_dist:.4f}")

    # One local search step
    if local_step_button:
        new_path, new_dist, improved = local_search_step(
            st.session_state.path_input, points, labels, label_to_index
        )
        st.session_state.path_input = new_path
        if improved:
            st.info(f"Local search step improved distance to {new_dist:.4f}")
        else:
            st.info("Local optimum reached: no improving adjacent swap found.")

    # Automatic local search until local optimum
    if local_auto_button:
        current_path = st.session_state.path_input
        info_placeholder = st.empty()

        max_iters = 1000
        for it in range(max_iters):
            new_path, new_dist, improved = local_search_step(
                current_path, points, labels, label_to_index
            )
            if not improved:
                info_placeholder.success(
                    f"Local search finished after {it} iterations. "
                    f"Local optimum distance: {compute_path_distance(current_path, points, label_to_index):.4f}"
                )
                break
            current_path = new_path
            st.session_state.path_input = current_path
            info_placeholder.info(
                f"Iteration {it+1}, distance = {new_dist:.4f}"
            )
            if step_time > 0:
                time.sleep(step_time)

    # Path input (bound to session_state)
    path_input = st.text_input(
        "Path (e.g. ABCD)",
        value=st.session_state.path_input,
        key="path_input",
        max_chars=100,
    )

    distance = None
    error_msg = None
    covers_all = False

    if path_input.strip():
        try:
            distance = compute_path_distance(path_input, points, label_to_index)
            covers_all = visits_all_nodes(path_input, labels)
        except ValueError as e:
            error_msg = str(e)

    # Show distance and "covers all nodes?" property
    if error_msg:
        st.error(error_msg)
    elif distance is not None:
        color = "green" if covers_all else "red"
        status_text = (
            "✅ Path visits all nodes."
            if covers_all
            else "❌ Path does NOT visit all nodes."
        )

        # Bigger distance text
        st.markdown(
            f"<p style='font-size: 26px; font-weight: bold; color:{color};'>"
            f"Distance: {distance:.4f}"
            f"</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<p style='font-size: 18px; color:{color};'>{status_text}</p>",
            unsafe_allow_html=True,
        )

# ---------- Center column: plot + distance matrix ----------

with col_center:
    st.subheader("Plot")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig = go.Figure()

    # Points + labels
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=10),
            name="Points",
        )
    )

    # Path in red (if letters are valid)
    path_to_draw = path_input.strip().upper()
    if path_to_draw and all(ch in label_to_index for ch in path_to_draw) and len(path_to_draw) >= 2:
        path_indices = [label_to_index[ch] for ch in path_to_draw]
        path_x = [points[i][0] for i in path_indices]
        path_y = [points[i][1] for i in path_indices]

        fig.add_trace(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode="lines+markers",
                line=dict(color="red", width=3),
                marker=dict(size=8),
                name="Path",
            )
        )

    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(range=[0, 10]),
        yaxis=dict(range=[0, 10]),
        width=800,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Distance matrix
    st.subheader("Distance matrix")
    dist_df = compute_distance_matrix(points, labels)
    st.dataframe(dist_df.style.format("{:.3f}"))
