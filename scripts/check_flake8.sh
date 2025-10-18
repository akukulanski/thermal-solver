#!/bin/bash
set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ${SCRIPT_DIR}/..
uv run flake8 ./thermal_solver --count --select=E9,F63,F7,F82 --show-source --statistics
