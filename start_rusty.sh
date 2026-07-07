#!/bin/bash

cd /home/rusty/Rusty-FER || exit 1
mkdir -p logs

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python src/app.py >> logs/rusty_autostart.log 2>&1
