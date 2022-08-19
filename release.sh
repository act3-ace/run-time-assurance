#!/bin/bash -e

# script looks for a variable __version__ in a python file and updates its value of the version
# Matches the examples:
# __version__ = "0.0.0"
# __version__ = '0.0.0'

NEXT_RELEASE="$1"
sed -i 's/^__version__\s*=\s*["'"'"'][[:digit:]]\+\.[[:digit:]]\+\.[[:digit:]]\+["'"'"']/__version__ = '"\"${NEXT_RELEASE}\"/" "version.py"
