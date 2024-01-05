#!/bin/bash -e
NEXT_RELEASE="$1"

# Update Poetry version if lock file exist
if [ -f poetry.lock ]; then
    poetry version "${NEXT_RELEASE}"
fi

# Update Version file
echo "${NEXT_RELEASE}" >VERSION
