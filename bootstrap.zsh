#!/usr/bin/env zsh

set -euo pipefail

echo "Resetting the project's virtual environment"
rm -rf .venv
python -m venv .venv

echo 'Installing packages'
. .venv/bin/activate
poetry install
