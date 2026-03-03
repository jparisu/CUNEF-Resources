#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
APP_DIR="$ROOT_DIR/streamlit_apps/logistic_regression_app"

python -m pip install --upgrade pip
python -m pip install -r "$APP_DIR/requirements.txt" pytest

python -m py_compile "$APP_DIR/app.py"
python -m py_compile "$APP_DIR/src/logistic_regression/metrics.py"

pytest -q "$APP_DIR/tests/test_cloud_compat.py"

echo "Cloud compatibility checks passed."
