#!/bin/bash -e
NEXT_RELEASE="$1"

# Update Poetry version if lock file exist
if [ -f poetry.lock ]; then
    poetry version "${NEXT_RELEASE}"
    poetry export -f requirements.txt -o requirements.dep.txt --with lint,test,docs
fi

# Update Version file
echo "${NEXT_RELEASE}" >VERSION
