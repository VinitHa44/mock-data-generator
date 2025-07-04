#!/bin/bash
# Delete all __pycache__ directories excluding dhiwise_3_0_.venv
find . -type d \( -name "__pycache__" -a ! -path "./.venv/*" \) -prune -exec rm -r {} +
# Run autoflake to remove unused imports in Python files excluding dhiwise_3_0_.venv
find . -type f -name "*.py" -not -path "./.venv/*" -exec autoflake --remove-all-unused-imports --in-place {} \;
# Run isort to sort imports in Python files excluding dhiwise_3_0_.venv
find . -type f -name "*.py" -not -path "./.venv/*" -exec isort {} \;
# Run black to format Python files with a line length of 120 characters excluding dhiwise_3_0_.venv
find . -type f -name "*.py" -not -path "./.venv/*" -exec black {} --line-length 80 \;
# Run vulture to find unused functions in Python files excluding dhiwise_3_0_.venv
vulture . --exclude .venv