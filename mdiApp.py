# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:05:59 2020

@author: coxan
"""


import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, uic
import cv2

from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, videoFile):
        super().__init__()
        self.videoFile = videoFile

    def run(self):
#        cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(self.videoFile)
        while True:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(int(w*0.9), int(h*0.9))#, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
            else: # Stop the running thread
                break


class VideoWidget(QWidget):
    def __init__(self, videoFile):
        super().__init__()
        self.top = 0
        self.left = 0
        self.width = 1280
        self.height = 720
        self.minwidth = 500
        self.minheight = 500
        self.videoFile = videoFile
        self.initUI()
        
    def __del__(self):
        self.th.terminate()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):
        self.setGeometry(self.left, self.top, self.width, self.height)
           # setting the minimum size 
        self.setMinimumSize(self.minwidth, self.minheight)
        #self.resize(1800, 1200)
        # create a label
        self.label = QLabel(self)
        self.label.setMargin(0)
        self.label.move(0, 0)
        self.label.resize(600, 600)
        self.th = Thread(self.videoFile)
        self.th.changePixmap.connect(self.setImage)
        self.th.start()
        self.show()


class MainWindow(QtWidgets.QMainWindow):
    count = 0

    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        bar = self.menuBar()
        
        file = bar.addMenu("Subwindow")
        file.addAction("New Video Window")
        file.addAction("New Window")
        file.addAction("Cascade")
        file.addAction("Tile")
        file.addAction("K360")
        file.triggered[QAction].connect(self.click)
        self.setWindowTitle("Multiple window using MDI")
        
    def click(self, q):

        if q.text() == "K360":
            print("Get the subvideo for K360")
            for subvideo in [i for i in os.listdir('./videos/NewYork_subvids')]:
                videoFilepath = os.path.join('./videos/NewYork_subvids', subvideo)
                MainWindow.count = MainWindow.count+1
                sub = QMdiSubWindow()
                sub.setWidget(VideoWidget(videoFilepath))
                sub.setWindowTitle(subvideo)
                self.mdi.addSubWindow(sub)
                sub.show()

        if q.text() == "New Video Window":
            print("New sub window")
            MainWindow.count = MainWindow.count+1
            sub = QMdiSubWindow()
            sub.setWidget(VideoWidget('./videos/NewYork.mp4'))
            sub.setWindowTitle("subwindow"+str(MainWindow.count))
            self.mdi.addSubWindow(sub)
            sub.show()

            
        if q.text() == "New Window":
            print("New text window")
            MainWindow.count = MainWindow.count+1
            sub = QMdiSubWindow()
            sub.setWidget(QTextEdit())
            sub.setWindowTitle("subwindow"+str(MainWindow.count))
            self.mdi.addSubWindow(sub)
            sub.show()
                    
        if q.text() == "Cascade":
            self.mdi.cascadeSubWindows()
                    
        if q.text() == "Tile":
            self.mdi.tileSubWindows()
		
def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
	
if __name__ == '__main__':
    main()
