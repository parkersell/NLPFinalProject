import os
import clip
import torch
from torchvision.datasets import CIFAR100
import json
from PIL import Image
import numpy as np
import pickle
import re

"""
Updated to make so that it stores each video under one file
Updated to make it into functions to allow for modifications to the algorithm
"""

CLIPFEATURES_PATH = '/mnt/d/CLIPFEATURES/'
FRAMES_PATH = '/mnt/d/TastyFrames/'

# Load the clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50x4', device)
   
# save clip_features pickle file
def clip_features_save(video_name: str, clip_features: dict) -> None :
    featurepath = CLIPFEATURES_PATH + video_name + '.pkl'
    print(f"Finished {video_name} with {len(clip_features)} frames")
    with open(featurepath, 'wb') as f:
        pickle.dump(clip_features, f)

# creates a clip weight with size (1, 640) from an image's filepath
def create_clip_from_image(image_filepath: str) -> np.array :
    with torch.no_grad(): # needed for model prediction
        image = Image.open(image_filepath).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return np.array(image_features.cpu()) # clip weight

def create_from_json(json_path: str) -> None :  
    """
    Loads from a json which has information about each image's filepath and the selected frames that are annotated
    input: json_path
    output: None, creates files in CLIPFEATURES FOLDER
    
    """

    # load the json with the cooresponding dictionary with caption and filepath
    imgs = json.load(open(json_path, 'r'))
    imgs = imgs['images']
    print(len(imgs))

    clip_features = {} # a dictionary to store all the clip features for each image for one video -> key: filepath, value: (1,640) numpy array 
    prev_video_name = None # the previous video_name

    # go through all the images in the json file
    for img in imgs:
       
        filepath = img['file_path']
        cur_video_name = re.search("\w+\/([\w-]+)",filepath).group(1)
        if not prev_video_name:
            print("Starting", cur_video_name)
            prev_video_name = cur_video_name
        
        # if we moved on to a new video, save the video dictionary, and restart dictionary
        if cur_video_name != prev_video_name:
            clip_features_save(prev_video_name, clip_features)
            clip_features = {}
            print("Starting", cur_video_name)
    
        clip_features[filepath] = create_clip_from_image(filepath)
        
        # give a progress update per 1000 frames
        if len(clip_features) % 1000 == 0:
            print(f"Frame No: {len(clip_features)} of {cur_video_name}")
        prev_video_name = cur_video_name

    # need to do last one since it always saves the i - 1 dictionary 
    clip_features_save(prev_video_name, clip_features)

def clipfeatures_V2():
    """
    Addresses two main issues with V1's method
        1. Filepath is incorrectly named with the full relative path -> Renames file_path to just the image number i.e. 00001.jpg
        2. Only selected frames are loaded, but this is not always the case -> Loads missing frames
    """

    # Issue 1 fix 
    def rename_filepaths(clip_features:dict, video_name: str): 
        "Renames filepaths to clip_features to a new clip_features dict"
        new_clip_features = {k.replace("../TastyFrames/"+ video_name + '/frames/', ''): v for k, v in clip_features.items()}
        print(f"Renamed {video_name}")
        return new_clip_features


    # Issue 2 fix 
    def add_missing_frames(clip_features:dict, video_name: str) -> None:
        "Adds missing frames to clip_features inplace"
        all_frames = os.listdir(FRAMES_PATH + video_name + '/frames/')
        all_frames.remove('finished.txt')

        selected_frames = clip_features.keys()
        
        missing_frames = all_frames - selected_frames

        for image in missing_frames:
            clip_features[image] = create_clip_from_image(FRAMES_PATH + video_name + '/frames/' + image)
        print(f"Added {len(missing_frames)} missing frames from {video_name}")


    all_videos = os.listdir(CLIPFEATURES_PATH)

    for video in all_videos:
        video_name = video.replace('.pkl', '')
        with open(CLIPFEATURES_PATH + video, 'rb') as f:
            clip_features = pickle.load(f)
        
        clip_features = rename_filepaths(clip_features, video_name)

        add_missing_frames(clip_features, video_name)

        clip_features_save(video_name, clip_features)


if __name__ == "__main__":
    
    # V1 method
    # create_from_json('clip/jsonAll.json')

    # second fixing method V1
    clipfeatures_V2()

