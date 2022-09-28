"""

Dataset Used : https://www.kaggle.com/datasets/omkargurav/face-mask-dataset


"""

import os
import shutil
from utils.CustomModel_Inference import CustomModel_Inference
from utils.CustomModelConfig import CustomModelConfig
import tensorflow as tf
from datetime import datetime

img_height = 300
img_width = 300
batch_size = 32

custom_model = CustomModel_Inference()
custom_model_config = CustomModelConfig()

ModelPath = os.path.join("models", "masknonmask", "VGG16_MaskNonMask.h5")
LabelPath = os.path.join("models", "masknonmask", "MaskNonMask_labels.txt")


custom_model.createVGG16Model()

now = datetime.now()
now = now.strftime("%m_%d_%Y_%H_%M")
DATA_PATH = "Data/with_without_mask"
OUTPUT_DIR = "outputs/{}/".format(now)
if not os.path.exists(OUTPUT_DIR):
	os.makedirs(OUTPUT_DIR)

custom_model_config.TRAIN_PATH = DATA_PATH 
custom_model_config.TEST_PATH= DATA_PATH
custom_model_config.OUTPUT_MODELS_PATH = OUTPUT_DIR + "MaskNonMask"
custom_model_config.LABELS_FILE_NAME =  "MaskNonMask_labels"

if not os.path.exists(custom_model_config.OUTPUT_MODELS_PATH):
	os.makedirs(custom_model_config.OUTPUT_MODELS_PATH)

custom_model_config.TRAIN_BATCH = 4
custom_model_config.EPOCHS = 2
custom_model_config.TRAINING_STEPS = 100
custom_model_config.VALIDATION_STEPS = 100
custom_model_config.PATIENCE = 10
custom_model_config.continue_training = False
custom_model_config.last_weight_file = ModelPath
custom_model_config.POST_MODEL_NAME = "VGG16_MaskNonMask"


custom_model.prepare_training(custom_model_config)

hist, modelName, selfmodel = custom_model.start_training(custom_model_config)


from matplotlib import pyplot as plt

try:    
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_accuracy'])
    plt.plot(hist.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'loss','val_acc', 'val_loss'], loc='upper left')
    plt.savefig("{}/{}_Discard_Images_ContainerDoor_VGG16.png".format(custom_model_config.OUTPUT_MODELS_PATH,now))
    plt.show()
except:
    pass