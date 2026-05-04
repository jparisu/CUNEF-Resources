import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st


Position = Tuple[int, int]
Move = str  # "up", "down", "left", "right"
DIRS: Dict[Move, Tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROWS = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
}
Evaluation = Tuple[int, Optional[int]]


@dataclass(frozen=True)
class GameState:
    n: int
    m: int
    p1: Position
    p2: Position
    lava: frozenset[Position]
    current_player: int  # 1 or 2


def initial_state(n: int, m: int) -> GameState:
    return GameState(
        n=n,
        m=m,
        p1=(0, 0),
        p2=(n - 1, m - 1),
        lava=frozenset(),
        current_player=1,
    )


def in_bounds(n: int, m: int, r: int, c: int) -> bool:
    return 0 <= r < n and 0 <= c < m


def occupied_by_other(state: GameState, player: int, pos: Position) -> bool:
    other = state.p2 if player == 1 else state.p1
    return pos == other


def legal_moves(state: GameState, player: Optional[int] = None) -> List[Move]:
    player = state.current_player if player is None else player
    pos = state.p1 if player == 1 else state.p2
    moves: List[Move] = []
    for move, (dr, dc) in DIRS.items():
        nr, nc = pos[0] + dr, pos[1] + dc
        nxt = (nr, nc)
        if not in_bounds(state.n, state.m, nr, nc):
            continue
        if nxt in state.lava:
            continue
        if occupied_by_other(state, player, nxt):
            continue
        moves.append(move)
    return moves


def apply_move(state: GameState, move: Move) -> GameState:
    player = state.current_player
    if move not in legal_moves(state, player):
        raise ValueError(f"Illegal move: {move}")

    dr, dc = DIRS[move]
    if player == 1:
        old = state.p1
        new_pos = (state.p1[0] + dr, state.p1[1] + dc)
        return GameState(
            n=state.n,
            m=state.m,
            p1=new_pos,
            p2=state.p2,
            lava=state.lava.union({old}),
            current_player=2,
        )

    old = state.p2
    new_pos = (state.p2[0] + dr, state.p2[1] + dc)
    return GameState(
        n=state.n,
        m=state.m,
        p1=state.p1,
        p2=new_pos,
        lava=state.lava.union({old}),
        current_player=1,
    )


def winner_if_terminal(state: GameState) -> Optional[int]:
    if legal_moves(state):
        return None
    return 2 if state.current_player == 1 else 1


def state_key(state: GameState) -> Tuple:
    return (
        state.n,
        state.m,
        state.p1,
        state.p2,
        tuple(sorted(state.lava)),
        state.current_player,
    )


class SearchLimitReached(Exception):
    pass


class MinimaxSolver:
    """
    Full minimax without pruning.

    For each state, memo stores a tuple:
        (outcome, plies)
    where, from the current player's perspective:
        outcome = +1 means current player can force a win
        outcome = -1 means current player loses with optimal play
        plies = number of half-moves until terminal position under optimal play

    Tie-breaking:
    - among winning moves, prefer the fastest win
    - among losing moves, prefer the slowest loss
    """

    def __init__(self, max_nodes: int = 100_000):
        self.max_nodes = max_nodes
        self.nodes = 0
        self.memo: Dict[Tuple, Evaluation] = {}

    def solve(self, state: GameState) -> Tuple[Optional[int], Dict[Move, Evaluation], bool, int]:
        self.nodes = 0
        self.memo.clear()
        moves = legal_moves(state)
        move_scores: Dict[Move, Evaluation] = {}

        try:
            if not moves:
                return winner_if_terminal(state), {}, True, self.nodes

            best_value: Optional[Evaluation] = None
            for mv in moves:
                child = apply_move(state, mv)
                child_outcome, child_plies = self._solve_state(child)
                value = (-child_outcome, child_plies + 1)
                move_scores[mv] = value
                if best_value is None or self._better_for_player(value, best_value):
                    best_value = value

            assert best_value is not None
            predicted_winner = state.current_player if best_value[0] > 0 else (2 if state.current_player == 1 else 1)
            return predicted_winner, move_scores, True, self.nodes
        except SearchLimitReached:
            return None, move_scores, False, self.nodes

    def evaluate_state(self, state: GameState) -> Tuple[int, int]:
        return self._solve_state(state)

    def _solve_state(self, state: GameState) -> Tuple[int, int]:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise SearchLimitReached()

        key = state_key(state)
        if key in self.memo:
            return self.memo[key]

        moves = legal_moves(state)
        if not moves:
            self.memo[key] = (-1, 0)
            return (-1, 0)

        best_value: Optional[Tuple[int, int]] = None
        for mv in moves:
            child = apply_move(state, mv)
            child_outcome, child_plies = self._solve_state(child)
            value = (-child_outcome, child_plies + 1)
            if best_value is None or self._better_for_player(value, best_value):
                best_value = value

        assert best_value is not None
        self.memo[key] = best_value
        return best_value

    @staticmethod
    def _better_for_player(candidate: Tuple[int, int], current: Tuple[int, int]) -> bool:
        cand_outcome, cand_plies = candidate
        curr_outcome, curr_plies = current

        if cand_outcome != curr_outcome:
            return cand_outcome > curr_outcome

        if cand_outcome > 0:
            return cand_plies < curr_plies

        return cand_plies > curr_plies


class PrunedMinimaxSolver:
    """
    Minimax with alpha-beta pruning.

    This solver only computes the winner under optimal play and one
    recommended move. It does not compute plies to finish.
    """

    def __init__(self, max_nodes: int = 100_000):
        self.max_nodes = max_nodes
        self.nodes = 0
        self.memo: Dict[Tuple, int] = {}

    def solve(self, state: GameState) -> Tuple[Optional[int], Optional[Move], bool, int]:
        self.nodes = 0
        self.memo.clear()
        moves = legal_moves(state)

        try:
            if not moves:
                return winner_if_terminal(state), None, True, self.nodes

            best_outcome: Optional[int] = None
            best_move: Optional[Move] = None
            alpha = -1
            beta = 1

            for mv in moves:
                child = apply_move(state, mv)
                child_outcome = self._solve_state(child, -beta, -alpha)
                outcome = -child_outcome
                if best_outcome is None or outcome > best_outcome:
                    best_outcome = outcome
                    best_move = mv
                alpha = max(alpha, outcome)
                if alpha >= beta:
                    break

            assert best_outcome is not None
            predicted_winner = state.current_player if best_outcome > 0 else (2 if state.current_player == 1 else 1)
            return predicted_winner, best_move, True, self.nodes
        except SearchLimitReached:
            return None, None, False, self.nodes

    def evaluate_state(self, state: GameState) -> int:
        return self._solve_state(state, -1, 1)

    def _solve_state(self, state: GameState, alpha: int, beta: int) -> int:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise SearchLimitReached()

        key = state_key(state)
        if key in self.memo:
            return self.memo[key]

        moves = legal_moves(state)
        if not moves:
            self.memo[key] = -1
            return -1

        best_outcome = -1
        for mv in moves:
            child = apply_move(state, mv)
            child_outcome = self._solve_state(child, -beta, -alpha)
            outcome = -child_outcome
            best_outcome = max(best_outcome, outcome)
            alpha = max(alpha, outcome)
            if alpha >= beta:
                break

        self.memo[key] = best_outcome
        return best_outcome


def draw_board(state: GameState, plot_size: float = 6.0):
    fig_size = max(2.0, min(6.0, float(plot_size)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    for r in range(state.n):
        for c in range(state.m):
            pos = (r, c)
            color = "#f4d03f"  # yellow
            if pos in state.lava:
                color = "#e74c3c"  # red
            if pos == state.p1:
                color = "#2ecc71"  # green
            if pos == state.p2:
                color = "#3498db"  # blue

            rect = Rectangle((c, state.n - 1 - r), 1, 1, facecolor=color, edgecolor="black")
            ax.add_patch(rect)

    ax.set_xlim(0, state.m)
    ax.set_ylim(0, state.n)
    ax.set_aspect("equal")
    ax.set_xticks(range(state.m + 1))
    ax.set_yticks(range(state.n + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_title("LavaFloor")
    fig.tight_layout()
    return fig


def reset_game(n: int, m: int):
    st.session_state.state = initial_state(n, m)
    st.session_state.history = []
    st.session_state.minimax_result = None


def apply_ui_move(move: Move):
    current = st.session_state.state
    st.session_state.history.append(current)
    st.session_state.state = apply_move(current, move)
    st.session_state.minimax_result = None


def undo_move():
    if st.session_state.history:
        st.session_state.state = st.session_state.history.pop()
        st.session_state.minimax_result = None


def get_active_minimax_result(state: GameState, solver_mode: str, max_nodes: int) -> Optional[dict]:
    minimax_info = st.session_state.minimax_result
    if minimax_info is None:
        return None
    if minimax_info.get("state_key") != state_key(state):
        return None
    if minimax_info.get("solver_mode") != solver_mode:
        return None
    if minimax_info.get("max_nodes") != int(max_nodes):
        return None
    return minimax_info


def get_button_type(move: Move, winning_moves: Set[Move]) -> str:
    return "primary" if move in winning_moves else "secondary"


def compute_live_minimax_annotations(
    state: GameState,
    minimax_info: Optional[dict],
) -> Tuple[Set[Move], Dict[Move, str], Optional[int], Optional[Tuple[int, int]]]:
    legal = set(legal_moves(state))
    highlighted_moves: Set[Move] = set()
    move_labels: Dict[Move, str] = {mv: ARROWS[mv] for mv in DIRS}

    if not minimax_info or not minimax_info.get("complete"):
        return highlighted_moves, move_labels, None, None

    if minimax_info["solver_mode"] == "full":
        solver: MinimaxSolver = minimax_info["solver"]
        try:
            state_eval = solver.evaluate_state(state)
        except SearchLimitReached:
            return highlighted_moves, move_labels, None, None

        state_outcome, _ = state_eval
        current_winner = state.current_player if state_outcome > 0 else (2 if state.current_player == 1 else 1)

        for mv in legal:
            child = apply_move(state, mv)
            try:
                child_outcome, child_plies = solver.evaluate_state(child)
            except SearchLimitReached:
                continue
            outcome, plies = -child_outcome, child_plies + 1
            move_labels[mv] = f"{ARROWS[mv]} ({plies})"
            if current_winner == state.current_player and outcome > 0:
                highlighted_moves.add(mv)

        return highlighted_moves, move_labels, current_winner, state_eval

    best_move = minimax_info.get("best_move")
    if best_move in legal:
        highlighted_moves.add(best_move)
    return highlighted_moves, move_labels, minimax_info.get("predicted_winner"), None


def main():
    st.set_page_config(page_title="LavaFloor", layout="wide")
    st.title("LavaFloor")

    if "state" not in st.session_state:
        st.session_state.state = initial_state(4, 4)
    if "history" not in st.session_state:
        st.session_state.history = []
    if "minimax_result" not in st.session_state:
        st.session_state.minimax_result = None
    if "use_pruning" not in st.session_state:
        st.session_state.use_pruning = False

    with st.sidebar:
        st.header("Configuration")
        with st.expander("Board settings", expanded=True):
            n = st.number_input("N (rows)", min_value=2, max_value=12, value=st.session_state.state.n, step=1)
            m = st.number_input("M (columns)", min_value=2, max_value=12, value=st.session_state.state.m, step=1)
            if st.button("New game / Reset", use_container_width=True):
                reset_game(int(n), int(m))
                st.rerun()

        with st.expander("Display settings", expanded=True):
            plot_size = st.slider(
                "Board plot size",
                min_value=2.0,
                max_value=6.0,
                value=4.0,
                step=0.5,
                help="Resize the board plot.",
            )

        with st.expander("Minimax settings", expanded=True):
            use_pruning = st.toggle(
                "Use alpha-beta pruning",
                value=st.session_state.use_pruning,
                help=(
                    "Off: full minimax computes winner and half-moves to finish. "
                    "On: alpha-beta pruning computes the winner and one recommended move only."
                ),
            )
            st.session_state.use_pruning = use_pruning
            max_nodes = st.number_input(
                "MAX_NODES",
                min_value=1_000,
                max_value=10_000_000,
                value=100_000,
                step=10_000,
                help="Maximum number of nodes explored by minimax before stopping.",
            )

    state: GameState = st.session_state.state
    terminal_winner = winner_if_terminal(state)

    board_col, control_col = st.columns([2.3, 1.1])

    solver_mode = "pruned" if use_pruning else "full"
    minimax_info = get_active_minimax_result(state, solver_mode, int(max_nodes))
    highlighted_moves, move_labels, current_winner, state_eval = compute_live_minimax_annotations(state, minimax_info)
    legal = set(legal_moves(state))

    with board_col:
        fig = draw_board(state, plot_size=plot_size)
        st.pyplot(fig, clear_figure=True)

        if terminal_winner is None:
            st.subheader(f"Next player: P{state.current_player}")
        else:
            st.subheader(f"Game over. Winner: P{terminal_winner}")

        st.markdown("### Legend")
        st.markdown(
            "- Green: P1\n"
            "- Blue: P2\n"
            "- Red: Lava\n"
            "- Yellow: Empty tile"
        )

    with control_col:
        st.markdown("### Controls")
        up_col_l, up_col_c, up_col_r = st.columns([1, 1, 1])
        with up_col_c:
            if st.button(
                move_labels["up"],
                key="move_up",
                disabled=("up" not in legal) or (terminal_winner is not None),
                use_container_width=True,
                type=get_button_type("up", highlighted_moves),
            ):
                apply_ui_move("up")
                st.rerun()

        mid_col_l, mid_col_c, mid_col_r = st.columns([1, 1, 1])
        with mid_col_l:
            if st.button(
                move_labels["left"],
                key="move_left",
                disabled=("left" not in legal) or (terminal_winner is not None),
                use_container_width=True,
                type=get_button_type("left", highlighted_moves),
            ):
                apply_ui_move("left")
                st.rerun()
        with mid_col_c:
            if st.button("Undo", key="undo_btn", disabled=not st.session_state.history, use_container_width=True):
                undo_move()
                st.rerun()
        with mid_col_r:
            if st.button(
                move_labels["right"],
                key="move_right",
                disabled=("right" not in legal) or (terminal_winner is not None),
                use_container_width=True,
                type=get_button_type("right", highlighted_moves),
            ):
                apply_ui_move("right")
                st.rerun()

        down_col_l, down_col_c, down_col_r = st.columns([1, 1, 1])
        with down_col_c:
            if st.button(
                move_labels["down"],
                key="move_down",
                disabled=("down" not in legal) or (terminal_winner is not None),
                use_container_width=True,
                type=get_button_type("down", highlighted_moves),
            ):
                apply_ui_move("down")
                st.rerun()

        st.markdown("### Minimax")
        if st.button("Calculate Minimax", use_container_width=True, disabled=(terminal_winner is not None)):
            if solver_mode == "full":
                solver = MinimaxSolver(max_nodes=int(max_nodes))
                predicted_winner, move_scores, complete, nodes = solver.solve(state)
                st.session_state.minimax_result = {
                    "predicted_winner": predicted_winner,
                    "move_scores": move_scores,
                    "complete": complete,
                    "nodes": nodes,
                    "max_nodes": int(max_nodes),
                    "solver": solver,
                    "solver_mode": solver_mode,
                    "state_key": state_key(state),
                }
            else:
                solver = PrunedMinimaxSolver(max_nodes=int(max_nodes))
                predicted_winner, best_move, complete, nodes = solver.solve(state)
                st.session_state.minimax_result = {
                    "predicted_winner": predicted_winner,
                    "best_move": best_move,
                    "complete": complete,
                    "nodes": nodes,
                    "max_nodes": int(max_nodes),
                    "solver": solver,
                    "solver_mode": solver_mode,
                    "state_key": state_key(state),
                }
            st.rerun()

        minimax_info = get_active_minimax_result(state, solver_mode, int(max_nodes))
        if minimax_info is not None:
            if minimax_info["complete"]:
                st.success(
                    f"Optimal-play winner from current position: P{minimax_info['predicted_winner']} "
                    f"(searched {minimax_info['nodes']} nodes)."
                )
                if solver_mode == "full" and current_winner == state.current_player:
                    good = [move_labels[mv] for mv in legal if mv in highlighted_moves]
                    if good:
                        st.info("Winning move(s) highlighted: " + ", ".join(good))
                if solver_mode == "pruned" and minimax_info.get("best_move") in legal:
                    label = ARROWS[minimax_info["best_move"]]
                    if minimax_info["predicted_winner"] == state.current_player:
                        st.info(f"Recommended winning move highlighted: {label}")
                    else:
                        st.info(f"Best available move highlighted: {label}")
                if state_eval is not None:
                    outcome, plies = state_eval
                    status = "winning" if outcome > 0 else "losing"
                    st.caption(f"Current position is {status} for P{state.current_player} in {plies} half-moves with optimal play.")
            else:
                st.warning(
                    f"Minimax stopped after reaching MAX_NODES = {minimax_info['max_nodes']:,}. "
                    f"Explored {minimax_info['nodes']:,} nodes. Result is partial."
                )
                if solver_mode == "full" and minimax_info["move_scores"]:
                    partial = ", ".join(
                        f"{ARROWS[mv]} ({plies}): {'win' if outcome > 0 else 'loss'}"
                        for mv, (outcome, plies) in minimax_info["move_scores"].items()
                    )
                    st.caption("Partial move evaluations: " + partial)


if __name__ == "__main__":
    main()
