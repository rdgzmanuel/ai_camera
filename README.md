# ai_camera
Repository containing the code for developing an automated camera control system.


## Folder structure
The folder /ai_camera is the main folder of the repository. It contains the rest of the required files and directories:
    - /models contains the downloaded models for object detection. Link to model download page: https://github.com/ultralytics/ultralytics

    - /videos contains the input videos in which to perform object detection. Also contains the downloaded and prepared soccertrack
    dataset for model training.

    - .gitignore includes the folders that are ignored by git when pushing to the repo. These are large and heavy folders that are better
    kept on lccal devices.

    - .yaml file includes the specification for the yolo finetuning process.

    - /src contains the source code and the __init__.py file. Not all fles are used, some of them where developed for trials or testing.

## How to run?
From ai_camera, run python -m src.<name_of_the_file_without_extension> to run that file, like python -m src.object_detector

To obtain the datasets to train models, the prepare_ files are used. Each of them specifies what kind of data is prepared.

motion_detector.py and object_detector.py are the files used for object detection, bia open_cv or yolo, respectively.

Install the necessary libraries using pip install -r requirements.txt

To start using the code, two repositories need to be cloned at the same level of ai_camera (outside it):

### ByteTrack ###
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
python setup.py develop


 Only if it was not installed with the requirements:

### YOLOX ###
git clone https://github.com/Megvii-BaseDetection/YOLOX
cd YOLOX
pip install -v -e .

Two files neede to be fixed: replace dtype=np.float by dtype=float in YOLO/yolox/tracker/matching.py and byte_tracker.py
