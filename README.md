# Tello Gesture Controller

<p align="center">
  <img src="https://user-images.githubusercontent.com/47962267/164318346-0f7472a1-cc3d-45fe-a5e2-0b56163f3057.gif">
</p>

1. [**Setup & Installation**](https://github.com/briancsavage/Tello-Hand-Signal-Controller#setup-and-installation)
2. [**Repo Structure**](https://github.com/briancsavage/Tello-Hand-Signal-Controller/blob/main/README.md#repo-structure)
3. [**Usage Steps**](https://github.com/briancsavage/Tello-Hand-Signal-Controller#usage-steps)
4. [**Implementation Steps**](https://github.com/briancsavage/Tello-Hand-Signal-Controller#implementation-steps)
5. [**Resources & References**](https://github.com/briancsavage/Tello-Hand-Signal-Controller#resources-and-references)

<br>

```
Steps Still Todo

1. [ ] Make script to handle data generation and training processes
2. [ ] Collect more training data from drone webcam
3. [ ] Scale dataset to multiple via randomly adding noise and varying training examples
4. [ ] Swap object detector for hand landmark detector?
5. [ ] ???
```

<br>

## Setup and Installation
🟢 ***With Weights*** 🟢
1. Clone repository using `git clone https://github.com/briancsavage/Tello-Hand-Signal-Controller.git`
2. Navigate to repository using `cd Tello-Hand-Signal-Controller`
3. Activate virtual environment via `. venv/Scripts/activate` in the root of the repo directory.
4. Install the required dependencies via `pip install -r requirements.txt` in the activated environment.
5. To run inference from the webcam run `python /src/router.py`, this will pull image data from `webcam(0)` on the system.

<br>

🟠 ***With Data*** 🟠
1. Our goal is to train the `ssd_mobilenet_v2` model using the TensorFlow training script and our hand signal training and testing data
2. Assuming you already have the repository locally, perform steps 2-4 of the ***With Weights*** section above to activate the virtual environment and install the necessary dependencies
3. The only dependency that isn't completeled handled by `pip` is `TensorFlow Object Detection API`, and thus, we need to follow the setups steps [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation). 
   * The directory, `TensorFlow` should be place in the same parent directory as the `Tello-Hand-Signal-Controller` directory (i.e. adjacent directories).
5. Use the provided, `recognizer.py` to call the training script from within the `TensorFlow Object Detection API`

<br>

🔴 ***Without Either*** 🔴
1. Our goal is to collect training data via the webcam of the computer and then once we have our labeled data, we can complete the previous step titled, ***With Data*** and train up the sign detection model.
2. Use the provided, `recognizer.py` to pull image data from the computer webcam, displays current label being collected to console with instructions on how to construct the hand signal, and saves the image into the data directory within the subdirectory titled images (i.e. `Tello-Hand-Signal-Controller/data/images`)
3. Use a LabelImg to annotate the collected images with each of the hand signal operations the drone should recognize (i.e. `up`, `down`, ...)
4. Goto the **With Data** section above and complete steps to train the `ssd-mobilenet-v2` with the labelled object detection image data.

<br>

## Repo Structure

```
data/
│
├── images/
│   ├── ...
│   ├── up-abc123.jpg
│   ├── up-def456.jpg
│   └── up-ghi789.jpg
│
├── annotations/
│   ├── ...
│   ├── up/
│   └── down/
│
├── records/
│   ├── train.record
│   └── test.record
│
├── train/
│   ├── ...
│   ├── up-abc123.jpg
│   └── up-abc123.xml
│
└── test/
    ├── ...
    ├── up-def456.jpg
    └── up-def456.xml
    
src/
│
├── utils/
│   └── generate_tfrecord.py
│
├── main.py
├── controller.py
├── detector.py
├── recognizer.py
└── router.py

models/
│
├── face-detection/
│   └── landmark_weights.pt
│
└── ssd-mobilenet-v2/
    ├── ...
    ├── ckpt-7.index
    ├── ckpt-7.data-000-of-001
    ├── ckpt-8.index
    └── ckpt-8.data-000-of-001

venv/
│
└── ...

setup.py
requirements.txt
README.txt
```

<br>

## Usage Steps
***After necessary dependency installation and training***
```bash
python /src/router.py
```
<br>

## Implementation Steps
1. `TODO in final report`

<br>

## Resources and References
1. [TensorFlow Object Detection Tutorial](https://www.youtube.com/watch?v=pDXdlXlaCco&t=475s)
2. [TensorFlow XML-to-TFRecord Converter](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py)
3. [TensorFlow Record Format Specs](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details)
