from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
from PIL import ImageFont
import time
from scipy.stats import norm
import cv2

# delete after
from sensor_msgs.msg import Image as image_msg
import rospy
from cv_bridge import CvBridge

class TLClassifier(object):
    def __init__(self):

        self.GRAPH_FILE = 'light_classification/model/frozen_inference_graph.pb'
        self.COLOR_LIST =[(255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 255, 0)]
        self.COLOR_TEXT =['white', 'red', 'yellow', 'green']

        self.detection_graph = self.load_graph(self.GRAPH_FILE)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.states = [TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN]

        self.session = tf.Session(graph=self.detection_graph)
        self.image_pub = rospy.Publisher('image_traffic', image_msg, queue_size=1)
        self.bridge = CvBridge()

    def get_classification(self, image):
        image_expand = np.expand_dims(image, axis=0)
        (boxes, scores, classes) = self.session.run([self.detection_boxes, self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: image_expand})
        (scores, classes) = self.session.run([self.detection_scores, self.detection_classes], feed_dict={self.image_tensor: image_expand})


        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.8
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
        # classes = self.filter_boxes2(confidence_cutoff, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.

        height, width, _ = image.shape
        box_coords = self.to_image_coords(boxes, height, width)

        # Each class with be represented by a differently colored box
        self.draw_boxes(image, box_coords, classes, scores)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, encoding="rgb8"))
        if len(classes) > 0:
            c_index = int(classes[0]) - 1


            return self.states[c_index]
        else:
            return TrafficLight.UNKNOWN


    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes


    def filter_boxes2(self, min_score, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_classes = classes[idxs, ...]
        return filtered_classes


    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords


    def draw_boxes(self, image, boxes, classes, scores, thickness=2):
        """Draw bounding boxes on the image"""
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            # if class_id > 2:
            #     print(class_id)
            color = self.COLOR_LIST[class_id]
            color_text = self.COLOR_TEXT[class_id]

            cv2.rectangle(image, (left, top), (right, bot), color, thickness)

            fnt = cv2.FONT_HERSHEY_SIMPLEX
            txt = color_text + " "+ str(round(scores[i],2))

            text_rect = cv2.getTextSize(txt, fnt, 1, 2)[0]

            cv2.rectangle(image, (left, top), (int(left + text_rect[0]), int(top + text_rect[1])), color, -1)
            cv2.rectangle(image, (left, top), (right, bot), color, thickness)
            cv2.putText(image, txt, (left, int(top + 20)), fnt, 1, (0,0,0), 2)


    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)

                tf.import_graph_def(od_graph_def, name='')

        return graph
