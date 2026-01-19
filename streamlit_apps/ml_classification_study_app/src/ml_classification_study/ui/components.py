from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from ml_classification_study.types import ArgSpec


def render_arg_specs(
    specs: List[ArgSpec],
    *,
    key_prefix: str,
    initial: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Render Streamlit widgets for an ArgSpec list.

    Returns a dict mapping spec.key -> value.
    """

    initial = initial or {}
    values: Dict[str, Any] = {}

    for spec in specs:
        widget_key = f"{key_prefix}:{spec.key}"
        default = initial.get(spec.key, spec.default)

        if spec.kind == "bool":
            values[spec.key] = st.checkbox(spec.label, value=bool(default), key=widget_key, help=spec.help or None)
            continue

        if spec.kind == "choice":
            if not spec.choices:
                raise ValueError(f"ArgSpec '{spec.key}' is choice but has no choices")
            labels = [lbl for (lbl, _val) in spec.choices]
            vals = [_val for (_lbl, _val) in spec.choices]
            try:
                idx = vals.index(default)
            except ValueError:
                idx = 0
            chosen_label = st.selectbox(spec.label, options=labels, index=idx, key=widget_key, help=spec.help or None)
            values[spec.key] = vals[labels.index(chosen_label)]
            continue

        if spec.kind in ("int", "float"):
            min_v = spec.min_value
            max_v = spec.max_value
            step = spec.step

            # Use number_input for very small steps (sliders get awkward)
            use_number_input = (step is not None and step < 1e-4) or (min_v is None or max_v is None)

            if spec.kind == "int":
                if use_number_input:
                    values[spec.key] = int(
                        st.number_input(
                            spec.label,
                            value=int(default),
                            step=int(step) if step is not None else 1,
                            key=widget_key,
                            help=spec.help or None,
                        )
                    )
                else:
                    values[spec.key] = int(
                        st.slider(
                            spec.label,
                            min_value=int(min_v),
                            max_value=int(max_v),
                            value=int(default),
                            step=int(step) if step is not None else 1,
                            key=widget_key,
                            help=spec.help or None,
                        )
                    )
            else:
                if use_number_input:
                    values[spec.key] = float(
                        st.number_input(
                            spec.label,
                            value=float(default),
                            step=float(step) if step is not None else 0.1,
                            key=widget_key,
                            help=spec.help or None,
                            format="%.12g",
                        )
                    )
                else:
                    values[spec.key] = float(
                        st.slider(
                            spec.label,
                            min_value=float(min_v),
                            max_value=float(max_v),
                            value=float(default),
                            step=float(step) if step is not None else 0.1,
                            key=widget_key,
                            help=spec.help or None,
                        )
                    )
            continue

        if spec.kind == "text":
            values[spec.key] = st.text_input(spec.label, value=str(default), key=widget_key, help=spec.help or None)
            continue

        raise ValueError(f"Unknown ArgSpec kind: {spec.kind}")

    return values


def plotly_chart_stretch(container: Any, fig: Any, **kwargs: Any) -> Any:
    """Render a Plotly chart that stretches to the container width.

    Streamlit deprecated `use_container_width` in favor of `width="stretch"`.
    This helper prefers the new API and falls back to the old one for
    compatibility with older Streamlit versions.
    """

    try:
        return container.plotly_chart(fig, width="stretch", theme=None, **kwargs)
    except TypeError:
        # Older Streamlit versions.
        return container.plotly_chart(fig, use_container_width=True, theme=None, **kwargs)


def button_stretch(container: Any, label: str, **kwargs: Any) -> bool:
    """Render a full-width button.

    Streamlit deprecated `use_container_width` in favor of `width="stretch"`.
    """

    try:
        return bool(container.button(label, width="stretch", **kwargs))
    except TypeError:
        return bool(container.button(label, use_container_width=True, **kwargs))
