from moviepy.editor import *

#convert from mov to mp4
# input_file = 'raw_mom.mov'
# output_file = 'raw_mom.mp4'
# video = VideoFileClip(input_file)
# video.write_videofile(output_file, codec='libx264')
# video.close()


myclip = VideoFileClip("python/animation.mp4")

newclip = (myclip.fx(vfx.colorx, 1.2))

newclip.write_videofile("output_video.mp4")
