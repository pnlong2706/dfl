#!/bin/bash
set -xe
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Check if a username was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: ./run.sh <your_unique_username>"
  exit 1
fi

echo "--- Running script for user: $1 ---"
USER="$1" python app/main.py
echo "--- Script finished ---"
