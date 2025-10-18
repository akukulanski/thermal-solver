#!/bin/bash
set -eu

grep -rin --color -E "FIXME|TODO" src/
