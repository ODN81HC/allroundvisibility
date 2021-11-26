# Perfom cutting a small specific part of a lengthy video
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
start_time = 1
end_time = 300
ffmpeg_extract_subclip("./videos/overview_1.MP4", start_time, end_time, targetname="./videos/overview_output.mp4")