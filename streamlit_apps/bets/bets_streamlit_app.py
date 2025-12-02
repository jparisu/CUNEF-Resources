import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import random
from pathlib import Path


def compute_bet_payoff_array(bet_type: str, b_value: int, x_values: np.ndarray) -> np.ndarray:
    """
    For a single bet, compute the payoff for each possible R in x_values.

    bet_type:
        "high" -> payoff = R - b
        "low"  -> payoff = b - R
        "none" -> payoff = 0
    """
    if bet_type == "high":
        return x_values - b_value
    elif bet_type == "low":
        return b_value - x_values
    else:  # "none"
        return np.zeros_like(x_values, dtype=int)


def main():
    st.set_page_config(
        page_title="Betting Strategy Visualizer",
        layout="wide",
    )

    st.title("Betting Strategy Visualizer")
    st.caption(
        "Explore the outcomes for different **high / low / none** bets "
        "when a random number R is uniform in [1, S]."
    )

    # ============================
    # HELP / README BUTTON
    # ============================
    readme_text = None
    readme_path = Path(__file__).parent / "bets_readme.md"
    if readme_path.exists():
        try:
            readme_text = readme_path.read_text(encoding="utf-8")
        except Exception:
            readme_text = None

    if "show_help" not in st.session_state:
        st.session_state["show_help"] = False

    if readme_text is not None:
        if st.button("❓ Show help / instructions"):
            # Toggle help visibility
            st.session_state["show_help"] = not st.session_state["show_help"]

        if st.session_state["show_help"]:
            st.markdown(readme_text)
            st.markdown("---")
    else:
        st.info("Help file `bets_readme.md` not found in the app directory.")

    # Layout: LEFT (info), CENTER (plots), RIGHT (parameters)
    left_col, center_col, right_col = st.columns([1.2, 2.0, 1.2])

    # ============================
    # RIGHT COLUMN – PARAMETERS
    # ============================
    with right_col:
        st.header("Parameters")

        st.subheader("Global settings")
        S = st.slider(
            "Size of range S (numbers 1 to S)",
            min_value=1,
            max_value=200,
            value=20,
            step=1,
        )

        num_bets = st.number_input(
            "Number of bets",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )
        num_bets = int(num_bets)

        # Colors for bets – consistent across all plots
        base_colors = px.colors.qualitative.Plotly

        # Handle S changing so bet positions stay valid
        if "prev_S" not in st.session_state:
            st.session_state["prev_S"] = S

        if S != st.session_state["prev_S"]:
            for i in range(num_bets):
                key = f"bet_pos_{i}"
                if key in st.session_state:
                    st.session_state[key] = min(st.session_state[key], S)
            st.session_state["prev_S"] = S

        st.subheader("Bets (positions and action)")
        bets = []

        for i in range(num_bets):
            color = base_colors[i % len(base_colors)]

            # Colored header for this bet
            st.markdown(
                f"""
                <div style="
                    padding:0.25rem 0.5rem;
                    border-radius:0.5rem;
                    background-color:rgba(0,0,0,0.02);
                    border-left:0.4rem solid {color};
                    margin-top:0.5rem;
                    margin-bottom:0.2rem;">
                    <strong style="color:{color}">Bet {i+1}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            pos = st.slider(
                f"Position b\u2096 (bet {i+1})",  # bᵢ
                min_value=1,
                max_value=S,
                step=1,
                key=f"bet_pos_{i}",
            )

            bet_type = st.radio(
                f"Action for bet {i+1}",
                options=["high", "low", "none"],
                horizontal=True,
                key=f"bet_type_{i}",
            )

            bets.append(
                {
                    "index": i + 1,
                    "position": pos,
                    "type": bet_type,
                    "color": color,
                }
            )

    # ==================================
    # CORE COMPUTATION (INVISIBLE AREA)
    # ==================================
    x_values = np.arange(1, S + 1)  # all possible R values
    total_points = np.zeros_like(x_values, dtype=int)

    for bet in bets:
        y = compute_bet_payoff_array(bet["type"], bet["position"], x_values)
        bet["payoff_array"] = y
        total_points += y

    # Expected value, min, max over uniform R in [1, S]
    if S > 0:
        mean_points = float(np.mean(total_points))
        max_points = int(np.max(total_points))
        min_points = int(np.min(total_points))
    else:
        mean_points = 0.0
        max_points = 0
        min_points = 0

    # ============================
    # LEFT COLUMN – SUMMARY + R
    # ============================
    with left_col:
        st.header("Outcome summary")

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean points (expected value)", f"{mean_points:.2f}")
        m2.metric("Maximum points over R", f"{max_points}")
        m3.metric("Minimum points over R", f"{min_points}")

        st.markdown("---")
        st.subheader("Try a specific value R")

        # Keep R in session_state so the random button & slider play nicely
        if "selected_R" not in st.session_state:
            st.session_state["selected_R"] = 1

        if st.session_state["selected_R"] > S:
            st.session_state["selected_R"] = S

        random_button = st.button("🎲 Pick R randomly")
        if random_button and S >= 1:
            st.session_state["selected_R"] = random.randint(1, S)

        selected_R = st.slider(
            "Current selected value R",
            min_value=1,
            max_value=max(S, 1),
            key="selected_R",
        )

        points_for_selected = int(total_points[selected_R - 1])
        st.metric(f"Total points if R = {selected_R}", f"{points_for_selected}")

        with st.expander("Show contribution of each bet for this R"):
            for bet in bets:
                contribution = int(bet["payoff_array"][selected_R - 1])
                sign = "+" if contribution >= 0 else ""
                st.markdown(
                    f"<span style='color:{bet['color']};'>"
                    f"<strong>Bet {bet['index']} ({bet['type']})</strong>: "
                    f"{sign}{contribution}"
                    f"</span>",
                    unsafe_allow_html=True,
                )

    # ========================
    # CENTER COLUMN – PLOTS
    # ========================
    with center_col:
        st.header("Visualizations")

        # ---------- Plot 1: Line 1..S with bet points ----------
        fig1 = go.Figure()

        # Base line of integers 1..S
        fig1.add_trace(
            go.Scatter(
                x=x_values,
                y=np.zeros_like(x_values),
                mode="lines+markers",
                line=dict(width=1),
                marker=dict(size=4),
                name="Values 1..S",
                hoverinfo="x",
            )
        )

        # Points for each bet (colored)
        for bet in bets:
            fig1.add_trace(
                go.Scatter(
                    x=[bet["position"]],
                    y=[0],
                    mode="markers+text",
                    marker=dict(size=14, color=bet["color"]),
                    text=[f"b{bet['index']}"],
                    textposition="top center",
                    name=f"Bet {bet['index']} ({bet['type']})",
                    hovertemplate=(
                        "Bet %{text}<br>b = %{x}<extra></extra>"
                    ),
                )
            )

        fig1.update_layout(
            height=220,
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_title="Value R",
            yaxis_title="",
            yaxis=dict(showticklabels=False, zeroline=False),
            title="Bet positions along the line 1..S",
        )

        st.plotly_chart(fig1, use_container_width=True)

        # ---------- Plot 2: One line per bet ----------
        fig2 = go.Figure()

        for bet in bets:
            fig2.add_trace(
                go.Scatter(
                    x=x_values,
                    y=bet["payoff_array"],
                    mode="lines",
                    name=f"Bet {bet['index']} ({bet['type']})",
                    line=dict(width=2, color=bet["color"]),
                    hovertemplate=(
                        "R = %{x}<br>"
                        "Points = %{y}<br>"
                        f"<extra>Bet {bet['index']}</extra>"
                    ),
                )
            )

        fig2.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=40),
            xaxis_title="Value R",
            yaxis_title="Points from a single bet",
            title="Points from each bet as R varies",
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ---------- Plot 3: Total points bar chart ----------
        bar_colors = ["#888888"] * len(x_values)
        selected_idx = selected_R - 1
        if 0 <= selected_idx < len(bar_colors):
            bar_colors[selected_idx] = "#222222"  # highlight chosen R

        fig3 = go.Figure(
            data=[
                go.Bar(
                    x=x_values,
                    y=total_points,
                    marker=dict(color=bar_colors),
                    hovertemplate="R = %{x}<br>Total points = %{y}<extra></extra>",
                )
            ]
        )

        fig3.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=30, b=40),
            xaxis_title="Value R",
            yaxis_title="Total points (sum over all bets)",
            title="Total points for each possible R",
        )

        st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
