import openfabmap_python3 as of

# configuration for vocabulary
SETTINGS = dict()
SETTINGS["VocabTrainOptions"] = dict()
SETTINGS["VocabTrainOptions"]["ClusterSize"] = 0.45

vb = of.VocabularyBuilder(SETTINGS)

# use OpenCV cpp methods for image loading and feature extraction
png_file = "lenna.png"
vb.load_and_add_training_image(png_file)

# pass image as numpy array
from PIL import Image

img = Image.open(png_file)
vb.add_training_image(img)

# pass already-extracted features
import cv2

gray = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create(nfeatures=1500)
keypoints, descriptors = orb.detectAndCompute(gray, None)
vb.add_training_descs(descriptors)
