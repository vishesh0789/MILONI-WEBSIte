#!/bin/bash
# Catan Analysis Board launcher for macOS (double-click in Finder) and Linux (./start.command).
# First run: creates a private Python environment in .venv and installs what it needs.
# Extra options are passed through, e.g.  ./start.command --me red   or   ./start.command --resume
cd "$(dirname "$0")" || exit 1

pick_python() {
  for p in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      echo "$p"; return 0
    fi
  done
  return 1
}

if [ ! -x .venv/bin/python ]; then
  PY=$(pick_python) || {
    echo "Python 3.10 or newer is needed. Install it with:  brew install python@3.12"
    echo "(or from https://www.python.org/downloads/), then run this again."
    read -r -p "Press Enter to close." _; exit 1
  }
  echo "First run: setting up a Python environment with $PY ..."
  "$PY" -m venv .venv || { read -r -p "Could not create .venv. Press Enter to close." _; exit 1; }
fi

# (Re)install requirements when they change.
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  echo "Installing requirements (one time) ..."
  .venv/bin/python -m pip install --quiet --upgrade pip >/dev/null 2>&1
  if .venv/bin/python -m pip install --quiet -r requirements.txt; then
    touch .venv/.installed
  else
    echo "Warning: some packages failed to install. The board still works; reading screenshots needs opencv."
  fi
fi

echo "Starting the Catan Analysis Board. Keep this window open while you play (Ctrl+C to stop)."
exec .venv/bin/python -m catan.gui_server "$@"
