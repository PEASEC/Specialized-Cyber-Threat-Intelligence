#!/bin/sh

version=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$version" ]]
then
    echo "No Python!"

else 
    parsedVersion=$(echo "${version//./}")
    if [[ "$parsedVersion" -gt "370" ]]
    then 
        echo "Valid version"
    else
        echo "Invalid version"
    fi
fi

# 1. Installing python
# 2. Installing cuda, cuDNN, pytorch
# 3. Installing requirements
# 4. Starting python script 
