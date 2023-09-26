import os
import subprocess

# Define the directory containing the videos
video_directory = '/localhome/zla247/projects/data/Visualizations'

# Iterate over each file in the directory
for filename in os.listdir(video_directory):
    # Check if the file is an .mp4 video
    if filename.endswith(".mp4"):
        # Define the full path to the video and the path for the output gif
        video_path = os.path.join(video_directory, filename)
        gif_path = os.path.join(video_directory, filename.replace(".mp4", ".gif"))

        # Use ffmpeg to convert the video to gif
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', 'fps=10,scale=320:-1:flags=lanczos',
            '-c:v', 'gif',
            gif_path
        ]

        subprocess.run(cmd)

print("Conversion completed!")
