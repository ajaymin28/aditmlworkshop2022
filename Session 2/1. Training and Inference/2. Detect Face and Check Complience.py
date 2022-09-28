from utils.MediaPipeFaceDetection import MediaPipeFaceDetection
import os

from utils.CustomModel_Inference import CustomModel_Inference

custom_model = CustomModel_Inference()



ModelPath = os.path.join("models", "masknonmask", "VGG16_MaskNonMask.h5")
LabelPath = os.path.join("models", "masknonmask", "MaskNonMask_labels.txt")


custom_model.initVGG16FromFullModel(ModelPath, LabelPath)


import cv2

InputFrameWindowName = "InputFrame Window"
OutputFrameWindowName = "OutputMaskNonMask"
WebCamViewWindowName = "WebCam Window" 
MediaPipeWindowName  = "Media Pipe Face Detection"

GreenColor = (0,255,0) #bgr
RedColor = (0,0,255)

cv2.namedWindow(InputFrameWindowName)

MediaPipeFace = MediaPipeFaceDetection()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if success:

        if image is not None:
            MediaPipeFace.setInput(image.copy())

            FaceLocations = MediaPipeFace.getFaceLocations()

            for faceId, faceLoc  in FaceLocations.items():

                x = faceLoc["x"]
                y = faceLoc["y"]
                w = faceLoc["w"]
                h = faceLoc["h"]

                cropimg = image[y:y+h,x:x+w]
                label,accuracy = custom_model.inferModel(cropimg)

                ColorToUse = GreenColor
                if label=="with_mask":
                    ColorToUse = GreenColor
                else:
                    ColorToUse = RedColor

                cv2.rectangle(image, (x, y), (x + w, y + h), ColorToUse, 2)
                cv2.putText(image, f"Predicted:{label}", (x-20,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ColorToUse, 2)


            cv2.imshow(OutputFrameWindowName, image)

    if cv2.waitKey(5) & 0xFF == 27:
        MediaPipeFace.exit_thread()   
        break

cv2.destroyAllWindows()
