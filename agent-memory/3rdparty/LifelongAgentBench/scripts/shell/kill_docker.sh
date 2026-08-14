#!/usr/bin/env bash

# Usage: ./kill_containers.sh [TIME_THRESHOLD]
# Example: ./kill_containers.sh 600  (kills containers created within the last 600 seconds)

TIME_THRESHOLD=${1:-300}  # Default to 300 if no argument is provided

now=$(date +%s)

for container in $(docker ps -aq); do
    created=$(docker inspect -f '{{.Created}}' "$container")
    created_timestamp=$(date -d "$created" +%s)
    age=$((now - created_timestamp))

    if [ "$age" -lt "$TIME_THRESHOLD" ]; then
        docker kill "$container"
    fi
done
