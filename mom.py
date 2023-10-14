from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import glob

def find_newest_file(folder_path):
    list_of_files = glob.glob(os.path.join(folder_path, '*.MOV'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def split_into_10s(input_folder, output_folder):
    newest_file = find_newest_file(input_folder)

    clip = VideoFileClip(newest_file)

    # Get the duration of the video in seconds
    duration = clip.duration

    # Calculate the number of 10-second clips
    num_clips = int(duration // 10)

    # Create MOM10s folder if it doesn't exist
    output_folder_path = os.path.join(os.getcwd(), output_folder)
    os.makedirs(output_folder_path, exist_ok=True)

    # Split the video into 10-second clips
    for i in range(num_clips):
        start_time = i * 10
        end_time = (i + 1) * 10
        subclip = clip.subclip(start_time, end_time)
        subclip.write_videofile(os.path.join(output_folder_path, f"clip_{i + 1}.mp4"), codec="libx264")

    clip.reader.close()
    clip.audio.reader.close_proc()

if __name__ == "__main__":
    input_folder = "MOMents"
    output_folder = "MOM10s"
    split_into_10s(input_folder, output_folder)