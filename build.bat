pyinstaller --onefile --clean --paths .\src --paths C:\Python27\Lib\site-packages  --add-data assets\*;assets --add-binary C:\Python27\Lib\site-packages\cv2\opencv_ffmpeg320_64.dll;. .\src\pcb_region_from_video.py 


copy .\doc\presentation.pptx .\dist\
copy .\video_demo.bat .\dist\