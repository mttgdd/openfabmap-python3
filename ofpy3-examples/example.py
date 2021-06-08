import openfabmap_python3 as of

# configuration for vocabulary
SETTINGS = dict()
SETTINGS["VocabTrainOptions"] = dict()
SETTINGS["VocabTrainOptions"]["ClusterSize"] = 0.45

SETTINGS["FeatureOptions"] = dict()
SETTINGS["FeatureOptions"]["FastDetector"]  = dict()
SETTINGS["FeatureOptions"]["FastDetector"]["Threshold"] = 10
SETTINGS["FeatureOptions"]["FastDetector"]["NonMaxSuppression"] = True
SETTINGS["FeatureOptions"]["SurfDetector"] = dict()
SETTINGS["FeatureOptions"]["SurfDetector"]["numFeatures"] = 0
SETTINGS["FeatureOptions"]["SurfDetector"]["nOctaveLayers"] = 3
SETTINGS["FeatureOptions"]["SurfDetector"]["contrastThreshold"] = 0.04 
SETTINGS["FeatureOptions"]["SurfDetector"]["edgeThreshold"] = 10
SETTINGS["FeatureOptions"]["SurfDetector"]["sigma"] = 1.6

vb = of.VocabularyBuilder(SETTINGS)
vb.init_detector_extractor(SETTINGS)

# use OpenCV cpp methods for image loading and feature extraction
png_file = "lenna.png"
vb.load_and_add_training_image(png_file)

# pass image as numpy array
from PIL import Image

img = Image.open(png_file)
vb.add_training_image(img)

# pass already-extracted features
import cv2
import numpy as np

vb = of.VocabularyBuilder(SETTINGS)

gray = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create(nfeatures=1500)
keypoints, descriptors = orb.detectAndCompute(gray, None)
descs = np.asarray(descriptors)
vb.add_training_descs(descs)
