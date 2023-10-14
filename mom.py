import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.video.io.VideoFileClip import VideoFileClip
import glob
import cv2
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def find_newest_file(folder_path):
    list_of_files = glob.glob(os.path.join(folder_path, '*.MOV'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_top_objects(frame_results, top_n=3, confidence_threshold=0.8):
    filtered_results = [obj for obj in frame_results if obj['confidence'] > confidence_threshold]
    sorted_results = sorted(filtered_results, key=lambda x: x['confidence'], reverse=True)
    return sorted_results[:top_n]

def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    images = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_interval == 0:
            images.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    list_of_lists_of_things = []

    for i, image in enumerate(images):
        model = YOLO("yolov8n.pt")
        results = model.predict(source=image, save_conf=True)
        confs = results[0].boxes.conf
        things = [model.names[int(c)] for i,c in enumerate(results[0].boxes.cls) if float(confs[i])>0.5]
        print(things)

        # make raw_image the object u want
        raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt")

        out = blip_model.generate(**inputs)
        print(processor.decode(out[0], skip_special_tokens=True))

        list_of_lists_of_things.append(things)
     
    return list_of_lists_of_things

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
        subclip.write_videofile(os.path.join(output_folder_path, f"1_clip_{i + 1}.mp4"), codec="libx264", threads=4)

    clip.reader.close()
    clip.audio.reader.close_proc()

if __name__ == "__main__":
    input_folder = "MOMents"
    output_folder = "MOM10s"
    #split_into_10s(input_folder, output_folder)
    extract_frames("MOM10s/clip_13.mp4")
