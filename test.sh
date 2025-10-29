#!/bin/bash
trap "kill 0" EXIT

for i in {1..10}; do
  python serverless.py &
done

wait
