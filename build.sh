#!/bin/bash
# Clean up previous build
rm -r ./dist

# Run the installer
pyinstaller --onefile ./src/pcb_region_from_video.py \
    --clean \
    --paths ./src \

# Copy assets to dist
mkdir dist/assets
cp ./assets/* ./

# Create launcers
echo "./pcb_region_from_video  assets/tracking1.MOV --step-through-frame --display-all" > ./dist/video_demo.sh
echo "./pcb_region_from_video  --from-cam" > ./dist/live_demo.sh
chmod +x ./dist/video_demo.sh
chmod +x ./dist/live_demo.sh