from elevenlabs import set_api_key,generate,save
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
import torch
from TTS.api import TTS

from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips


with open("eleven.pass", 'r') as file:
    set_api_key(file.read())


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_FILENAME = "tmp.jpg"

prompt = """
 have a 10 second video clip which has been proccessed by CV and ML stuff into this. 
 there is a transcript of what was said, and descriptions of what is detected by the yolo model and image captioner ever 2 seconds. 
 generate a cohesive funny story voiceover of what is happening to play over the 10s clip. it needs to be a very brief voiceover up to only 34 words
"""


with open("openai.pass", 'r') as file:
    openai.api_key = file.read()

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



def extract_frames(video_path, frame_interval=120):
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
    list_of_lists_of_things = []

    print(transcript)

    if len(transcript) < 40:
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

            face_id=None
            # face_id = id_face(Image.fromarray(image), "shrockers/")

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

def get_frames(folder):
    """
    Get a list of OpenCV images representing the first frame of each mp4 file in the specified folder.

    Parameters:
    - folder (str): The path to the folder.

    Returns:
    - list: A list of OpenCV images.
    """
    image_list = []

    try:
        # Get the list of files in the folder
        files = [f for f in os.listdir(folder) if f.endswith('.mp4')]

        # Iterate through each mp4 file
        for file in files:
            file_path = os.path.join(folder, file)

            # Open the video file
            cap = cv2.VideoCapture(file_path)

            # Read the first frame
            ret, frame = cap.read()

            # Append the first frame to the list
            if ret:
                image_list.append(frame)

            # Release the video capture object
            cap.release()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return image_list

def get_files(output_folder):
    # gpt_query = ""
    # for i,image in enumerate(get_frames(output_folder)):

    #     raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     face_id=None
    #     # face_id = id_face(Image.fromarray(image), "shrockers/")

    #     # unconditional image captioning
    #     inputs = processor(raw_image, return_tensors="pt")


    #     out = blip_model.generate(**inputs)
    #     desc = processor.decode(out[0], skip_special_tokens=True)
    #     gpt_query += f"{i}: {desc}\n"

    # gpt_query += "\n\n give me a comma separated list of numbers of the top 3 most interesting descriptions. DO NOT SAY ANYTHING ELSE EXCEPT THE ANSWER. YOUR ANSWER MUST BE FORMATTED LIKE THIS:  1,4,6"

    gpt_query = """0: a room with a carpet that has been painted blue
1: a man and woman sitting in a chair
2: a man in a suit and tie is standing in a room
3: a room with a wooden floor and a white wall
4: a man is walking through an office lobby
5: a man sitting in a chair with a laptop
6: a blur of people walking down a street
7: a man in a blue shirt is standing in front of a computer
8: a black and white rug
9: a door is open in an office
10: two people sitting in a room with computers
11: a man sitting on a chair in a room
12: a blur of a person walking down a street
13: a man sitting at a table in a restaurant
14: a man with a blue jacket
15: a man is walking up the stairs in a building
16: a blur of people on a train
17: a man holding a stuffed animal


 give me a comma separated list of numbers of the top 6 most interesting descriptions that make a good storyline together. DO NOT SAY ANYTHING ELSE EXCEPT THE ANSWER. YOUR ANSWER MUST BE FORMATTED LIKE THIS WHERE <number> is a number like 1 or 2 or 3 etc:  <number>,<number>,<number>,<number>,<number>,<number>
"""
    print(gpt_query)

    # Make a call to the OpenAI API
    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=gpt_query,
    max_tokens=1000
    )

    # Extract the generated message
    message = response['choices'][0]['text']
    print(message)

    numbers = [''.join([r for r in e if r in '1234567890']) for e in message.split(',')]
    files = [f"clip_{i}.mp4" for i in numbers if i]
    return files






if __name__ == "__main__":
    input_folder = "MOMents"
    output_folder = "MOM10s"
    #split_into_10s(input_folder, output_folder)

    # curr_clip = "MOM10s/clip_6.mp4"

    interesting_files = get_files(output_folder)

    clips = []
    for curr_clip in interesting_files:
        curr_clip = output_folder + "/" +curr_clip
        final_data = extract_frames(curr_clip)
        audio_clip = 0

        if len(final_data["transcript"]) < 40:
            # Make a call to the OpenAI API
            response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt + "\n" + str(final_data),
            max_tokens=200
            )

            # Extract the generated message
            message = response['choices'][0]['text']

            audio = generate(
            text=message,
            voice="Harry",
            model="eleven_multilingual_v2"
            )

            print(message)

            save(
                audio,  "output_eleven.wav"
            )

            audio_clip = AudioFileClip("output_eleven.wav")

            audio_clip = audio_clip.subclip(0, min([11,audio_clip.duration]))

        video_clip = VideoFileClip(curr_clip)
        if audio_clip:
            video_clip = video_clip.set_audio(audio_clip)

        clips.append(video_clip)
    
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile("running5.mp4")