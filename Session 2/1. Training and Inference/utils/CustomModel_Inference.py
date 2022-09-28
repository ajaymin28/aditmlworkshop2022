# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import time
from pathlib import Path
from PIL import Image


class CustomModel_Inference(object):
	"""docstring for CustomModel_Inference"""
	def __init__(self):
		self.ElapsedTime = 0
		self.classes = {}
		self.preTrainedModelLoaded = False
		print("CustomModel_Inference instance init (Done)")

	def initVGG16FromFullModel(self,modelPath,labelPath,saveWeightsName=""):
		self.modelPath = modelPath
		self.labelPath = labelPath
		
		print("Loading model from {}".format(modelPath))
		self.model = load_model(modelPath)
		if saveWeightsName!="":
			self.model.save_weights(saveWeightsName)
		print("Loading labels from {}".format(labelPath))
		self.load_labels(labelPath)

	def initVGG16FromWeightsFile(self,WeightsPath,labelPath):
		self.ElapsedTime = 0
		self.classes = {}
		print("Loading model from {}".format(WeightsPath))
		self.model = self.createVGG16Model()
		self.model.load_weights(WeightsPath)
		print("Loading labels from {}".format(labelPath))
		self.load_labels(labelPath)

	def createVGG16Model(self, nclasses=1,inputImage_shape=(300,300,3)):
		"""
		Creates VGG16 model with customized output layer with keras API
		"""
		self.outLayerNodes = 1
		self.lastLayer_ActivationFunction = "sigmoid"
		self.LossFunction = "binary_crossentropy"
		self.metrics_value = "accuracy"
		if nclasses>1:
			self.outLayerNodes = nclasses
			self.lastLayer_ActivationFunction = "softmax"
			self.LossFunction = "categorical_crossentropy"
			self.metrics_value = "accuracy"

		model_m = VGG16(weights='imagenet',include_top=False,input_shape=inputImage_shape)
		x = model_m.output
		global_average_layer = keras.layers.GlobalAveragePooling2D()
		x = global_average_layer(x)
		x = keras.layers.Dropout(0.2)(x)
		x = keras.layers.Flatten()(x)
		x = keras.layers.Dense(8, activation="relu")(x)
		predictions = keras.layers.Dense(self.outLayerNodes, activation=self.lastLayer_ActivationFunction)(x)
		self.model = keras.models.Model(model_m.input,predictions)

		opt = keras.optimizers.RMSprop(learning_rate=0.00001)
		self.model.compile(loss = self.LossFunction, optimizer = opt, metrics=[self.metrics_value])
		return self.model

	def load_labels(self,labelPath):
		with open(labelPath,'r') as f:
			lines = f.read()
			self.classes = eval(lines)

	def preprocessOpenCV_img(self,in_image):
		cvt_image = in_image[:,:,::-1]
		im_pil = Image.fromarray(cvt_image)
		im_resized = im_pil.resize((300, 300))
		img_data_main = image.img_to_array(im_resized)
		img_data = np.expand_dims(img_data_main, axis=0)
		img_data = preprocess_input(img_data)
		mat = np.array(img_data)
		return mat

	def preprocess_img(self,in_image_path):
		img = image.load_img(in_image_path, target_size=(300, 300))
		img_data_main = image.img_to_array(img)
		img_data = np.expand_dims(img_data_main, axis=0)
		img_data = preprocess_input(img_data)
		mat = np.array(img_data)
		return mat


	def inferModel(self,input_image,isBinaryModel=True, useImagePath=False):
		preprocessed_img = None
		if useImagePath:
			preprocessed_img = self.preprocess_img(input_image)
		else:
			preprocessed_img = self.preprocessOpenCV_img(input_image)

		if preprocessed_img is not None:
		
			start_time = time.clock()
			predictions = self.model.predict(preprocessed_img)
			self.ElapsedTime = time.clock() - start_time
			y_classes = None
			accuracies = []
			if isBinaryModel:
				y_classes = ((predictions > 0.5)+0).ravel()
				accuracies = self.calculateSigmoidPercentage(predictions)
			else:
				y_classes = predictions.argmax(axis=-1)
			label = self.classes[y_classes[0]]
			indexOf = self.getIndexOfLabel(label)
			return label,accuracies[indexOf]
		else:
			return None,None
	

	def calculateSigmoidPercentage(self,predictions,isBinaryModel=True):
		"""
		1. Sigmoid function gives prediction between 0 to 1 float values
		2. 0 to 0.5 values belong to classA (0 being high probability of classA)
		3. 0.5 to 1 values belong to classB (1 being high probability of classB)
		"""
		prob = float(predictions[0])
		prob = round(prob,2)
		probs = []
		if prob>0.5:
			prob = prob - 0.5
			classB = round(((prob*100)/0.5),2)
			classA = 100 - classB
			probs.append(classA)
			probs.append(classB)
		else:
			prob = 0.5 - prob 
			classA = round(((prob*100)/0.5),2)
			classB = 100 - classA
			probs.append(classA)
			probs.append(classB)
		return probs

	def getIndexOfLabel(self,label):
		index = 0
		for key, val in self.classes.items():
			if label==val:
				index = key
				break
		return index

	def prepare_training(self,TrainingConfig):

		train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
										rotation_range=TrainingConfig.rotation_range,
										shear_range=TrainingConfig.shear_range,
										horizontal_flip=TrainingConfig.horizontal_flip,
										width_shift_range=TrainingConfig.width_shift_range,
										height_shift_range=TrainingConfig.height_shift_range,
										zoom_range=TrainingConfig.zoom_range,
										validation_split=0.2)

		self.train_generator = train_datagen.flow_from_directory(TrainingConfig.TRAIN_PATH,
													target_size= TrainingConfig.TRAIN_IMAGE_SIZE,
													batch_size=TrainingConfig.TRAIN_BATCH,
													seed = TrainingConfig.seed,
													shuffle  = TrainingConfig.shuffle_data,
													class_mode=TrainingConfig.class_mode,
													subset="training"
													)

		self.validation_generator = train_datagen.flow_from_directory(TrainingConfig.TRAIN_PATH,
														target_size=TrainingConfig.TRAIN_IMAGE_SIZE,
														batch_size=TrainingConfig.TRAIN_BATCH,
														seed = TrainingConfig.seed,
														shuffle  = TrainingConfig.shuffle_data,
														class_mode=TrainingConfig.class_mode,
														subset="validation"
														)
		self.testing_generator = train_datagen.flow_from_directory(TrainingConfig.TEST_PATH,
														target_size=TrainingConfig.TRAIN_IMAGE_SIZE,
														batch_size=TrainingConfig.TRAIN_BATCH,
														seed = TrainingConfig.seed,
														shuffle  = TrainingConfig.shuffle_data,
														class_mode=TrainingConfig.class_mode,
														)
		
		labels = (self.train_generator.class_indices)
		labels = dict((v,k) for k,v in labels.items())
		self.classes = labels


		with open("{}/{}.txt".format(TrainingConfig.OUTPUT_MODELS_PATH,TrainingConfig.LABELS_FILE_NAME),'w') as f:
			f.write(str(labels))
			f.close()


		if TrainingConfig.continue_training:
			last_weight_file_check = Path(TrainingConfig.last_weight_file)
			if last_weight_file_check.exists():
				try:
					print("Loading pretrained weights from {}".format(TrainingConfig.last_weight_file))
					self.model.load_weights(TrainingConfig.last_weight_file)
					print("Loaded VGG weights, training will continue using provided weights file")
					self.preTrainedModelLoaded = True
				except Exception as e:
					print("Error loading vgg model {}".format(e))
			else:
				print("couldn't find last weight file, model will be trained from start")


	def start_training(self,TrainingConfig):

		es = EarlyStopping(monitor=TrainingConfig.EARLY_STOP_MONITOR, 
		mode=TrainingConfig.EARLY_STOP_MODE, 
		verbose=1, 
		patience=TrainingConfig.PATIENCE)
		
		Train_Samples = int(self.train_generator.samples)
		Validation_Samples = int(self.validation_generator.samples)


		TRAIN_STEPS_PER_EPOCH = np.ceil((Train_Samples/TrainingConfig.TRAIN_BATCH)-1)
		VALIDATION_STEPS_PER_EPOCH = np.ceil((Validation_Samples/TrainingConfig.TRAIN_BATCH)-1)


		hist = self.model.fit(self.train_generator,
                    steps_per_epoch= TRAIN_STEPS_PER_EPOCH,
                    epochs=TrainingConfig.EPOCHS,
                    validation_data = self.validation_generator,
                    validation_steps=VALIDATION_STEPS_PER_EPOCH,
                    callbacks=[es]
                    )


		modelName = TrainingConfig.OUTPUT_MODELS_PATH+ "/"+"{}{}".format(TrainingConfig.POST_MODEL_NAME,TrainingConfig.MODEL_EXTENSION)
		self.model.save(modelName)
		return hist, modelName, self.model