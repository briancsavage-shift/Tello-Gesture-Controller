import cv2
import os
import time
import uuid
import shutil
import random
import subprocess
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
from object_detection.builders import model_builder
from google.protobuf import text_format
from typing import Dict, List, Tuple, Any, Optional


class SignRecognizer:
    def __init__(self):
        #######################################################################
        #                                                                     #
        #                        TensorFlow Setup                             #
        #                                                                     #
        #######################################################################
        self.tensorflow_dir: os.path = \
            os.path.join(os.getcwd(), "..", "..", "TensorFlow")

        #######################################################################
        #                                                                     #
        #                        Data Labels Setup                            #
        #                                                                     #
        #######################################################################
        self.labels: Dict[str, Dict[str, Any]] = {
            "up": {
                "uses_both_hands": False,
                "description": "Pointer UP",
            },
            "down": {
                "uses_both_hands": False,
                "description": "Pointer DOWN",
            },
            "left": {
                "uses_both_hands": False,
                "description": "Pointer LEFT",
            },
            "right": {
                "uses_both_hands": False,
                "description": "Pointer RIGHT",
            },
            "forward": {
                "uses_both_hands": True,
                "description": "Parallel UP",
            },
            "backward": {
                "uses_both_hands": True,
                "description": "Parallel DOWN",
            },
            "clockwise": {
                "uses_both_hands": False,
                "description": "Pointer and Pinky UP, Thumb RIGHT",
            },
            "counterclockwise": {
                "uses_both_hands": False,
                "description": "Pointer and Pinky UP, Thumb LEFT",
            },
        }

        #######################################################################
        #                                                                     #
        #                        Directory Configs                            #
        #                                                                     #
        #######################################################################
        self.data_dir: os.path = os.path.join(os.getcwd(), "..", "data")
        self.annotations: os.path = os.path.join(self.data_dir, "annotations")
        self.records: os.path = os.path.join(self.data_dir, "records")
        self.images: os.path = os.path.join(self.data_dir, "images")

        self.tr_path: os.path = os.path.join(self.data_dir, "train")
        self.te_path: os.path = os.path.join(self.data_dir, "test")
        self.models_dir: os.path = os.path.join(os.getcwd(), "..", "models")
        self.model_config: os.path = \
            os.path.join(self.models_dir,
                         "ssd-mobilenet-v2",
                         "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config")

        #######################################################################
        #                                                                     #
        #                     Loading Model Checkpoint                        #
        #                                                                     #
        #######################################################################
        configs = config_util.get_configs_from_pipeline_file(self.model_config)
        self.detector = model_builder.build(model_config=configs['model'],
                                            is_training=False)
        self.checkpoint = tf.compat.v2.train.Checkpoint(model=self.detector)
        saved_models = os.path.join(self.models_dir,
                                        "ssd-mobilenet-v2",
                                        "saved_model")

        points = [f for f in os.listdir(saved_models) if f[:4] == "ckpt" and
                                                         f.endswith(".index")]
        latest = max([int(p.replace(".index", "").split("-")[-1])  # Version Num
                      for p in points])
        self.checkpoint.restore(os.path.join(self.models_dir,
                                             "ssd-mobilenet-v2",
                                             "saved_model",
                                             f"ckpt-{latest}")).expect_partial()

    def record_training_data(self, count_per_class: Optional[int] = 25) -> None:
        cv2.namedWindow("preview")
        cap: cv2.VideoCapture = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cv2.imshow("preview", frame)

        for label, info in self.labels.items():
            print(f"[ Collecting {count_per_class} images for {label} ]")
            print(f"**label:{label}** {info['description']} -> uses " +
                  ("both hands" if info['uses_both_hands'] else "either hand"))
            time.sleep(5)

            label_directory = os.path.join(self.images, label)
            if os.path.isdir(label_directory):
                for file in os.listdir(label_directory):
                    os.remove(os.path.join(label_directory, file))
            else:
                os.mkdir(label_directory)

            for i in range(count_per_class):
                print(f"\t[ Collecting image {i + 1} of {count_per_class} ]")
                image_path = os.path.join(label_directory,
                                          label + '-' +
                                          str(uuid.uuid4()) + '.jpg')
                ret, frame = cap.read()
                cv2.imwrite(image_path, frame)
                cv2.imshow("preview", frame)
                time.sleep(2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()

    @staticmethod
    def clear_directory(path: os.path,
                        recursive: Optional[bool] = False) -> None:
        if os.path.isdir(path):
            for file in os.listdir(path):
                if os.path.isfile(os.path.join(path, file)):
                    os.remove(os.path.join(path, file))
                elif recursive:
                    if os.path.isdir(os.path.join(path, file)):
                        SignRecognizer.clear_directory(path=os.path.join(path,
                                                                         file),
                                                       recursive=recursive)

    def build_directory(self, directory: os.path, filepaths: List[str]) -> None:
        for path in filepaths:
            filename = os.path.basename(path).split('.')[0]
            filepath = os.path.join(self.annotations, filename + '.xml')
            if os.path.isfile(filepath):
                jpg, xml = filename + '.jpg', filename + '.xml'
                shutil.copy(path, os.path.join(directory, jpg))
                shutil.copy(os.path.join(self.annotations, xml),
                            os.path.join(directory, xml))

    def shuffle_and_split(self, train_split: Optional[float] = 0.8) -> None:
        SignRecognizer.clear_directory(self.tr_path, recursive=False)
        SignRecognizer.clear_directory(self.te_path, recursive=False)
        images: List[Tuple[str, List[str]]] = []
        for label, info in self.labels.items():
            images.append((label,
                           [os.path.join(self.images, label, file)
                            for file in os.listdir(os.path.join(self.images,
                                                                label))]))
        for (label, image_paths) in images:
            tr_size: int = int(len(image_paths) * train_split)
            random.shuffle(image_paths)
            tr_paths: List[str] = image_paths[:tr_size]
            te_paths: List[str] = image_paths[tr_size:]
            self.build_directory(self.tr_path, tr_paths)
            self.build_directory(self.te_path, te_paths)

    def make_label_map(self) -> List[Dict[str, Any]]:
        labels_data: List[Dict[str, Any]] = []
        for i, (label, info) in enumerate(self.labels.items()):
            labels_data.append({"name": label, "id": i + 1})
        return labels_data

    def save_label_map(self, labels: List[Dict[str, Any]]) -> None:
        with open(os.path.join(self.annotations, 'label_map.pbtxt'), 'w') as f:
            for label in labels:
                f.write('item {\n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')

    def make_tf_records(self) -> None:
        script_path = os.path.join(os.getcwd(),
                                   "utils",
                                   "generate_tfrecord.py")
        command = lambda split: ["python", script_path,
                                 "-x", os.path.join(self.data_dir,
                                                    split),
                                 "-l", os.path.join(self.annotations,
                                                    "label_map.pbtxt"),
                                 "-o", os.path.join(self.records,
                                                    split + ".record")]
        for split in ["train", "test"]:
            subprocess.run(command(split), shell=True)

    def update_default_model_config(self):
        config_util.get_configs_from_pipeline_file(self.model_config)
        pipeline = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(self.model_config, 'r') as f:
            proto = f.read()
            text_format.Merge(proto, pipeline)
        pipeline.model.ssd.num_classes = 8
        pipeline.train_config.batch_size = 4
        pipeline.train_config.fine_tune_checkpoint = \
            os.path.join(self.models_dir, "ssd-mobilenet-v2", "ckpt-0")

        pipeline.train_config.fine_tune_checkpoint_type = "detection"
        pipeline.train_input_reader.label_map_path = \
            os.path.join(self.annotations, "label_map.pbtxt")

        pipeline.train_input_reader.tf_record_input_reader.input_path[:] = \
            [os.path.join(self.records, "train.record")]
        pipeline.train_input_reader.label_map_path = \
            os.path.join(self.annotations, "label_map.pbtxt")
        pipeline.eval_input_reader[0].tf_record_input_reader.input_path[:] = \
            [os.path.join(self.records, "test.record")]
        pipeline.eval_input_reader[0].label_map_path = \
            os.path.join(self.annotations, "label_map.pbtxt")

        save_config = text_format.MessageToString(pipeline)
        with tf.io.gfile.GFile(self.model_config, 'wb') as f:
            f.write(save_config)

    def train_model(self, train_steps: Optional[int] = 5000) -> None:
        model_path: str = str(os.path.join(self.models_dir,
                                           "ssd-mobilenet-v2",
                                           "saved_model"))
        config_path: str = str(self.model_config)
        command: List[str] = ["python", os.path.join(self.tensorflow_dir,
                                                     "models",
                                                     "research",
                                                     "object_detection",
                                                     "model_main_tf2.py"),
                              "--model_dir=" + model_path,
                              "--pipeline_config_path=" + config_path,
                              "--num_train_steps=" + str(train_steps)]
        subprocess.run(command, shell=True)

    @tf.function
    def detect(self, image):
        image, blobs = self.detector.preprocess(image)
        inference = self.detector.predict(image, blobs)
        detections = self.detector.postprocess(inference, blobs)
        return detections



if __name__ == "__main__":
    recognizer = SignRecognizer()
    # recognizer.record_training_data()
    # recognizer.shuffle_and_split()
    # labels = recognizer.make_label_map()
    # recognizer.save_label_map(labels)
    # recognizer.make_tf_records()
    # recognizer.update_default_model_config()
    # recognizer.train_model()

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    category_index = label_map_util.create_category_index_from_labelmap(
        os.path.join(recognizer.annotations, 'label_map.pbtxt'))

    while True:
        ret, frame = cap.read()
        image_np = np.array(frame)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),
                                            dtype=tf.float32)

        detections = recognizer.detect(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections[
            'detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        vis_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,
            min_score_thresh=0.5,
            agnostic_mode=False)

        cv2.imshow('object detection',
                   cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break