import sys
import cv2
import os
import glob
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class RGBHistogram:
    def __init__(self, bins):
        self.bins = bins
    def describe(self, image, mask=None):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    
#Main application class for the GUI
class PlantClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        #Set up the main window
        self.setWindowTitle("PLANT CLASSIFIER")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        #Label to display the selected image
        self.image_label = QLabel("Selected Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        #Label to display the classification result
        self.classification_label = QLabel("Prediction")
        self.classification_label.setAlignment(Qt.AlignCenter)
        #Label to display the accuracy of the model    
        self.accuracy_label = QLabel("Accuracy Model: N/A")
        self.accuracy_label.setAlignment(Qt.AlignCenter)
        #Button to select an image for classification
        self.select_button = QPushButton("Select Image")
        self.select_button.clicked.connect(self.select_image)
        #Button to train the model
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        #Layout for the central widget
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.classification_label)
        self.layout.addWidget(self.accuracy_label)
        self.layout.addWidget(self.select_button)
        self.layout.addWidget(self.train_button)
        self.central_widget.setLayout(self.layout)
        #Initialize variables for the model, descriptor, and label encoder
        self.model = None
        self.desc = None
        self.le = None
    #Function to select an image for classification
    def select_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg)")
        if image_path:
            self.predict_flower(image_path)
    #Function to predict the class of the selected image
    def predict_flower(self, image_path):
        if self.model is None or self.desc is None or self.le is None:
            QMessageBox.warning(self, "Error", "Model not trained yet.")
            return
        #Read the image and its mask (assuming mask is the same as image for now)
        image = cv2.imread(image_path)
        mask = cv2.imread(image_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #Extract features using the descriptor
        features = self.desc.describe(image, mask)
        #Predict the class of the image
        flower = self.le.inverse_transform(self.model.predict([features]))[0]
        #Display the image and the prediction
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaledToWidth(400)
        self.image_label.setPixmap(pixmap)
        self.classification_label.setText(f"I think this flower is a {flower.upper()}.")
    #Function to train the model
    def train_model(self):
        if self.model is not None:
            QMessageBox.warning(self, "Error", "Model has already been trained.")
            return
        #Select directories for images and masks
        images_dir = QFileDialog.getExistingDirectory(self, "Select Images Directory")
        masks_dir = QFileDialog.getExistingDirectory(self, "Select Masks Directory")
        #Load image and mask paths
        if not images_dir or not masks_dir:
            QMessageBox.warning(self, "Error", "Please select both images and masks directories.")
            return
        imagePaths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        maskPaths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        if len(imagePaths) == 0 or len(maskPaths) == 0:
            QMessageBox.warning(self, "Error", "No images or masks found. Check the paths.")
            return
        data = []
        target = []
        self.desc = RGBHistogram([8, 8, 8])
        #Process each image and corresponding mask
        for (imagePath, maskPath) in zip(imagePaths, maskPaths):
            image = cv2.imread(imagePath)
            if image is None:
                print(f"Error reading image {imagePath}")
                continue
            mask = cv2.imread(maskPath)
            if mask is None:
                print(f"Error reading mask {maskPath}")
                continue
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            #Extract features and append to the data list
            features = self.desc.describe(image, mask)
            data.append(features)
            #Extract the flower name from the filename
            filename = os.path.basename(imagePath)
            flower_name = filename.split("_")[0]
            target.append(flower_name)
        if len(data) == 0 or len(target) == 0:
            QMessageBox.warning(self, "Error", "No data to train. Check if images and masks are correctly loaded.")
            return
        targetNames = np.unique(target)
        self.le = LabelEncoder()
        target = self.le.fit_transform(target)
        #Split the data into training and testing sets
        (trainData, testData, trainTarget, testTarget) = train_test_split(data, target, test_size=0.3, random_state=42)
        #Train the model
        self.model = RandomForestClassifier(n_estimators=25, random_state=84)
        self.model.fit(trainData, trainTarget)
        #Calculate accuracy on the test set
        predictions = self.model.predict(testData)
        accuracy = accuracy_score(testTarget, predictions)
        self.accuracy_label.setText(f"Accuracy Model: {accuracy * 100:.2f}%")
        QMessageBox.information(self, "Success", "Model trained successfully.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlantClassifierApp()
    window.show()
    sys.exit(app.exec_())
