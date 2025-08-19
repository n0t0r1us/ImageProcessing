from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import glob
import cv2
import os

#Class for extracting RGB histogram features from images
class RGBHistogram:
    def __init__(self, bins):
        self.bins = bins
    #Compute a 3D histogram in the RGB color space and normalize it
    def describe(self, image, mask=None):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    
#Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to the image dataset")
ap.add_argument("-m", "--masks", required=True, help="path to the image masks")
args = vars(ap.parse_args())

#Load image and mask paths
imagePaths = sorted(glob.glob(os.path.join(args["images"], "*.jpg")))
maskPaths = sorted(glob.glob(os.path.join(args["masks"], "*.png")))

#Check if the image and mask are present
if len(imagePaths) == 0 or len(maskPaths) == 0:
    print("No images or masks found. Check the paths.")
    exit()
print(f"Found {len(imagePaths)} images and {len(maskPaths)} masks.")

data = []
target = []
desc = RGBHistogram([8, 8, 8]) #Initialize RGBHistogram with bins

#Extract features and labels from each image and mask pair
for (imagePath, maskPath) in zip(imagePaths, maskPaths):
    print(f"Processing image: {imagePath} and mask: {maskPath}")
    #Load the image
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Error reading image {imagePath}")
        continue
    #Load the mask
    mask = cv2.imread(maskPath)
    if mask is None:
        print(f"Error reading mask {maskPath}")
        continue
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) #Convert mask to grayscale
    #Extract features using RGBHistogram
    features = desc.describe(image, mask)
    data.append(features)
    #Extract the flower name correctly
    filename = os.path.basename(imagePath)
    flower_name = filename.split("_")[0]
    target.append(flower_name)

#Check if data was loaded correctly
if len(data) == 0 or len(target) == 0:
    print("No data to train. Check if images and masks are correctly loaded.")
    exit()

#Encode the target labels as integers
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

#Divide the data into training set and test set
(trainData, testData, trainTarget, testTarget) = train_test_split(data, target, test_size=0.3, random_state=42)

#Training the RandomForest model
model = RandomForestClassifier(n_estimators=25, random_state=84)
model.fit(trainData, trainTarget)

#Evaluate the model
print(classification_report(testTarget, model.predict(testData), target_names=targetNames))

#Predict and display 10 random images from the test set
for i in np.random.choice(np.arange(0, len(imagePaths)), 10):
    imagePath = imagePaths[i]
    maskPath = maskPaths[i]
    #Load the image and mask
    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #Extract features using the descriptor
    features = desc.describe(image, mask)
    #Predict the flower type
    flower = le.inverse_transform(model.predict([features]))[0]
    #Print the prediction and display the image
    print(imagePath)
    print("I think this flower is a {}".format(flower.upper()))
    cv2.imshow("image", image)
    cv2.waitKey(0)
