import sys
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar
from PyQt5.QtGui import QPixmap, QPainter, QPen, QCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image



# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Background worker for training
class TrainingWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def run(self):
        # Download dataset
        self.progress.emit(10)
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

        # Initialize model
        model = DigitClassifier().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 5
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Progress update per epoch
            self.progress.emit(20 + epoch * 15)

        # Save model
        torch.save(model.state_dict(), "mnist_model.pth")
        self.progress.emit(100)
        self.finished.emit()

# Main GUI
class DigitDrawingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handwritten Digit Classifier")
        self.setFixedSize(800, 450)
        self.setStyleSheet(self.getStyleSheet())

        self.canvas = QPixmap(200, 200)
        self.canvas.fill(Qt.black)

        self.label_canvas = QLabel(self)
        self.label_canvas.setPixmap(self.canvas)

        self.label_prediction = QLabel("Prediction: ?", self)
        self.label_prediction.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")

        self.btn_clear = QPushButton("Clear", self)
        self.btn_clear.setCursor((QCursor(Qt.CursorShape.PointingHandCursor)))
        self.btn_clear.clicked.connect(self.clearCanvas)

        self.btn_predict = QPushButton("Predict", self)
        self.btn_predict.setCursor((QCursor(Qt.CursorShape.PointingHandCursor)))
        self.btn_predict.clicked.connect(self.predictDigit)

        layout = QVBoxLayout()
        layout.addWidget(self.label_canvas)
        layout.addWidget(self.label_prediction)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_predict)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.drawing = False
        self.last_point = None

        # Load trained model
        self.model = DigitClassifier().to(device)
        self.model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
        self.model.eval()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.canvas)
            pen = QPen(Qt.white, 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            painter.end()
            self.last_point = event.pos()
            self.label_canvas.setPixmap(self.canvas)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clearCanvas(self):
        self.canvas.fill(Qt.black)
        self.label_canvas.setPixmap(self.canvas)
        self.label_prediction.setText("Prediction: ?")

    def predictDigit(self):
        image = self.canvas.toImage()
        buffer = image.bits().asarray(image.byteCount())
        img_array = np.array(buffer, dtype=np.uint8).reshape((200, 200, 4))[:, :, 0]

        img = Image.fromarray(img_array).resize((28, 28)).convert("L")
        transform = transforms.ToTensor()
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(img_tensor)
            prediction = output.argmax(1).item()

        self.label_prediction.setText(f"Prediction: {prediction}")

    def getStyleSheet(self):
        return """
            QWidget {
                background-color: #1E1E1E;
                color: white;
                font-family: 'Arial';
            }
            QPushButton {
                background-color: #29A19C;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #257F76;
            }
        """

# Splash Screen
class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Initializing AI Model")
        self.setFixedSize(800, 450)
        self.setStyleSheet("background-color: white;")

        self.label = QLabel("Setting up AI Model...", self)
        self.label.setStyleSheet("font-size: 16px; font-weight: bold; color: #444;")
        self.progress = QProgressBar(self)
        self.progress.setValue(0)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        self.setLayout(layout)

        self.worker = TrainingWorker()
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.launchMainApp)
        self.worker.start()

    def launchMainApp(self):
        self.close()
        self.main_app = DigitDrawingApp()
        self.main_app.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Check if model exists, otherwise train
    if not os.path.exists("mnist_model.pth"):
        splash = SplashScreen()
        splash.show()
    else:
        main_app = DigitDrawingApp()
        main_app.show()

    sys.exit(app.exec_())
