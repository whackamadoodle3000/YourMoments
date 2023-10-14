import os
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.video.io.VideoFileClip import VideoFileClip
import glob
import cv2
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import moviepy.editor as mp #for mov to mp3 conversion
from deepface import DeepFace
import PIL

DEFAULT_FILENAME = "tmp.jpg"


openai.api_key = "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"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def save_pil(image, filename = DEFAULT_FILENAME):
    image.save(filename)

'''
Takes in a PIL Image and path to database of named reference face images. 
Image names in the database should match names of the people.

Returns name of face if it finds one. 
'''
def id_face(pil_img, db_path):
    save_pil(pil_img)
    dfs = DeepFace.find(img_path = DEFAULT_FILENAME, db_path = db_path, enforce_detection=False)[0]
    if not dfs.shape[0]:
        return None
    return [dfs.sort_values("VGG-Face_cosine")["identity"][i].split("/")[-1][:-4] for i in range(dfs.shape[0])]


'''
Takes in a PIL Image.
Returns top emotion of face if there is a face. If there is no face, returns None.
'''
def id_emotion(pil_img):
    save_pil(pil_img)
    try:
        emotion = DeepFace.analyze(img_path = DEFAULT_FILENAME, actions = ['emotion'])
        return max(emotion[0]['emotion'].keys(), key = lambda k: emotion[0]['emotion'][k])
    except:
        return None # No face

def find_newest_file(folder_path):
    list_of_files = glob.glob(os.path.join(folder_path, '*.MOV'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def get_top_objects(frame_results, top_n=3, confidence_threshold=0.8):
    filtered_results = [obj for obj in frame_results if obj['confidence'] > confidence_threshold]
    sorted_results = sorted(filtered_results, key=lambda x: x['confidence'], reverse=True)
    return sorted_results[:top_n]

def extract_frames(video_path, frame_interval=30):
    # speech to text

    input_mov_file = video_path
    output_mp3_file = "raw_mom.mp3"
    video_clip = mp.VideoFileClip(input_mov_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_mp3_file, codec="mp3")
    video_clip.close()
    audio_clip.close()

    print(f"Conversion to MP3 complete. MP3 file saved as '{output_mp3_file}'.")


    #using whisper
    audio_file= open(output_mp3_file, "rb")
    transcript = (openai.Audio.transcribe("whisper-1", audio_file)).text

    print(transcript)

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

        # make raw_image the object u want
        raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_id = id_face(Image.fromarray(image), "shrockers/")

        # unconditional image captioning
        inputs = processor(raw_image, return_tensors="pt")


        out = blip_model.generate(**inputs)
        desc = processor.decode(out[0], skip_special_tokens=True)


        list_of_lists_of_things.append({"items":things, "description":desc, "face":face_id})
     
    final_data  = {"transcript" : transcript, "scene_descriptions":list_of_lists_of_things}
    print(final_data)
    return final_data

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



