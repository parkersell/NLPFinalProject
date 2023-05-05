This has 2 folders

- clip which is the the code that did holds the data pipeline for all the image data and text data with clip weights
    - Run to_jsonfile.py, and then testclip.py
- code_zero_shot which is the main model 
    - To train and run inference on the model:
        - First for the inital text recipe1m procedural understanding, run train_recipe1m.py
        - Using the checkpoint from that model, run train_videos.py 
        - Finally run inference_videosGT.py to test the text generation model on ground truth annotations given video input. 
        - Running inference_recipe1M.py tests the text generation model on just the recipe1m learned model with the previous step as input. 
    - To create the dataset from the raw xml files, run process_data.py to turn it all into a dictionary with annotation information
    - There are 3 pytorch dataset classes that use functions from the RECIPE1M.py and TastyVideos.py files
        - The vocabulary is a simple class to index the vocab. The dataloaders, then use nltk.word_tokenize to tokenize the text data.  

the get_frames_num.py is used to extract all the indices into one file for data preparation