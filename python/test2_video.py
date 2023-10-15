import moviepy.editor as mp
from moviepy.editor import *
import cv2
import numpy as np

#convert from mov to mp4
input_file = 'python/people_vid.mov'
output_file = 'python/people_vid.mp4'
video = VideoFileClip(input_file)
video.write_videofile(output_file, codec='libx264')
video.close()

# Load the video
video_clip = mp.VideoFileClip("python/people_vid.mp4")

def apply_frame_filter(frame, color, original_weight):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Apply color grading to the grayscale frame
    colored_frame = cv2.applyColorMap(gray_frame, color)
    # Combine the colored frame with the original frame
    result_frame = cv2.addWeighted(frame, original_weight, colored_frame, 1 - original_weight, 0)
    return result_frame


def apply_general_filter(video_clip, color, original_weight, lum, blackwhite): #lower lum num = darker
    #step1: tint
    if (not blackwhite):
        filtered_frames = [apply_frame_filter(frame, color, original_weight) for frame in video_clip.iter_frames(fps=video_clip.fps)]
        filtered_clip = mp.ImageSequenceClip(filtered_frames, fps=video_clip.fps) #add clips together to make vid
        filtered_clip = filtered_clip.set_audio(video_clip.audio) #fix audio
    
    else:
         filtered_clip = filtered_clip.fx(vfx.blackwhite)
    
    #step 2: luminosity
    filtered_clip = filtered_clip.fx(vfx.colorx, lum)

    filtered_clip.write_videofile("output_meow.mp4", codec='libx264')

    # Close the video clips
    video_clip.close()
    filtered_clip.close()


apply_general_filter(video_clip, cv2.COLORMAP_DEEPGREEN, 0.6, 1, False)


#1. spooky mystery y2k
#COLORMAP_DEEPGREEN, 0.6
#0.6
#false

#2. royal historical ball
#COLORMAP_OCEAN, 0.6
#0.3
#false

#3. dark night
#COLORMAP_BONE, 0.9
#0.3
#false

#4. faded memories
#COLORMAP_SUMMER, 0.6
#1
#false

#5. dirty yellow
#COLORMAP_DEEPGREEN, 0.9
#1
#false

#6. aesthetic film
#COLORMAP_VIRIDIS, 0.9
#0.6
#false

#7. spy movie noir, none
#none
#0.8
#true









#stuff i shoved into the function:
# filtered_clip = mp.ImageSequenceClip(filtered_frames, fps=video_clip.fps) #add clips together to make vid
# filtered_clip = filtered_clip.set_audio(video_clip.audio) #fix audio

# #lowering luminosity
# filtered_clip = filtered_clip.fx(vfx.colorx, 0.3)

# # Save the filtered video
# filtered_clip.write_videofile("output_meow.mp4", codec='libx264')

# # Close the video clips
# video_clip.close()
# filtered_clip.close()
