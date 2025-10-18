#!/bin/bash
set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ${SCRIPT_DIR}/..
uv run autopep8 --in-place --recursive ./src ./test
# -max-line-length=79
