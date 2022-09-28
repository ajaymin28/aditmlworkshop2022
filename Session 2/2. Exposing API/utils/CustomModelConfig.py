class CustomModelConfig:


	def __init__(self):

		self.TRAIN_PATH = ""
		self.TEST_PATH = ""
		self.LABELS_FILE_NAME = ""
		self.OUTPUT_MODELS_PATH = ""

		self.class_mode  = "binary"
		self.TRAIN_IMAGE_SIZE = (300,300)


		self.TRAIN_BATCH = 8
		self.EPOCHS = 20
		self.TRAINING_STEPS = 100
		self.VALIDATION_STEPS = 100

		"""
		Data Augmentation Options from ImageDataGenerator
		"""

		self.rotation_range=25
		self.shear_range=0.2
		self.horizontal_flip=True
		self.width_shift_range=0.2
		self.height_shift_range=0.2
		self.zoom_range=0.3

		self.shuffle_data = True
		self.seed = 1


		"""
		Training Loggers
		"""
		self.csv_logger_file = ""


		self.continue_training=False
		self.last_weight_file=""

		"""
		Default save model pre/post string to use while saving.
		"""  
		self.PRE_MODEL_NAME = ""
		self.POST_MODEL_NAME = "VGG16_CustomModel"
		self.MODEL_EXTENSION = ".h5"

		"""
		Early Stopping
		"""
		self.PATIENCE = 10
		self.EARLY_STOP_MONITOR = "val_loss"
		self.EARLY_STOP_MODE = "min"



	def config_to_dict(self):
		return {a: getattr(self, a) for a in sorted(dir(self)) if not a.startswith("__") and not callable(getattr(self, a))}

	def display_config(self):
		for key, val in self.config_to_dict().items():
			print(f"{key:30}: {val}")