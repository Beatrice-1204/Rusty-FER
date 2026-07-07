#!/bin/bash

pkill -INT -f "src/app.py"

sleep 2

if pgrep -f "src/app.py" > /dev/null; then
    pkill -TERM -f "src/app.py"
fi
