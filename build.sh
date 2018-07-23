#!/bin/bash


# Run the installer
pyinstaller --onefile \
    --clean \
    --paths ./src \
    --add-data assets/*:assets /
    ./src/pcb_region_from_video.py


# Create launcers
echo "./pcb_region_from_video  assets/tracking1.MOV --step-through-frame --display-all" > ./dist/video_demo.sh
echo "./pcb_region_from_video  --from-cam" > ./dist/live_demo.sh
chmod +x ./dist/video_demo.sh
chmod +x ./dist/live_demo.sh