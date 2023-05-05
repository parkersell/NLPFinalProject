import os
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
import cv2
from imutils.video import count_frames
from numpy import number

if __name__ == '__main__':

    recipe_names_fid = open("clip/ALL_RECIPES.txt", "r")
    recipe_names = recipe_names_fid.readlines()
    recipe_names_fid.close()

    path_2recipes = '/mnt/d/TastyVideos/'

    for indu in range(len(recipe_names)) :
        curr_recipe = recipe_names[indu].replace('\n', '')
        frames_path = '/mnt/d/TastyFrames/' + curr_recipe + '/frames/'
        print('video ', curr_recipe)

        curr_file_path = path_2recipes + curr_recipe
        video_file_path = curr_file_path + '/recipe_video.mp4'

        # Make the directory if it doesn't exist
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)

        # Prevents it from reextracting it
        if os.path.exists(frames_path + 'finished.txt'):
          with open(frames_path + 'finished.txt', 'r') as fp: 
            value = fp.read().strip()
            print('finished with', value)
            continue
          

        vidcap = cv2.VideoCapture(video_file_path)
        success,image = vidcap.read()
        count = 0
        while success:
          img_path = frames_path + "{:05d}".format(count) + '.jpg'
          image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
          cv2.imwrite(img_path, image)
          success,image = vidcap.read()
          count += 1
        if count != 0: 
          with open(frames_path + 'finished.txt', 'w') as fp:
            fp.write(str(count))