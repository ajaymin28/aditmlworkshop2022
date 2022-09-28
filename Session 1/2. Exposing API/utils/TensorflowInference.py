from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from io import BytesIO
import base64
import numpy as np
import cv2


class TensorflowInference:
    """
    Tensorflow Inference Handler
    """

    def __init__(self, modelPath, LabelsPath):
        self.Object_labels = {}
        # Load the model
        self.__loadModel(modelPath=modelPath)
        self.__loadLabels(LabelsPath=LabelsPath)


    def __loadLabels(self, LabelsPath):
        """
        Loads labels such as
        
        0 Blue Pen
        1 Green Pen

        To Dict

        {
            0: BluePen
            1: GrenPen
        }
        """  
        lines = []
        with open(LabelsPath) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            labels_substrings = line.split(" ")
            myStringLabel = "".join(labels_substrings[1:])
            self.Object_labels[int(labels_substrings[0])] = myStringLabel

    def __loadModel(self, modelPath):
        """
        Load Keras Model
        """
        self.model = load_model(modelPath)


    def infer(self, inputImage, isopencvImage=True):
        """
        Model Inference
        """

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        if not isopencvImage:
            inputImage = np.asarray(inputImage)

        image_resized = cv2.resize(inputImage, (224,224))
        normalized_image_array = (image_resized.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = self.model.predict(data)
        pred_max_index = np.argmax(prediction,axis=-1)[0]
        Pred_label = self.Object_labels[pred_max_index]
        PredConfidence = prediction[0][pred_max_index]

        return Pred_label, PredConfidence




