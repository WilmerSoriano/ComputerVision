
import sys
import numpy as np
import skvideo.io

import numpy
numpy.float = numpy.float64
numpy.int = numpy.int_

from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QProgressDialog
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen

from MotionDetector import MotionDetector

class GUI(QMainWindow):
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multi-Object Tracker")
        
        # Load video
        self.video = skvideo.io.vread(video_path)
        self.total_frames = self.video.shape[0]
        self.current_frame = 0
        
        # Precompute all tracking data upfront to avoid lag
        self.precomputed_tracking = self.precompute_tracking()
        
        # UI Elements
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_frames - 1)
        self.slider.valueChanged.connect(self.on_slider_change)
        
        self.btn_back60 = QPushButton("<< 60")
        self.btn_back60.clicked.connect(lambda: self.jump_frames(-60))
        self.btn_forward60 = QPushButton("60 >>")
        self.btn_forward60.clicked.connect(lambda: self.jump_frames(60))
        
        controls = QHBoxLayout()
        controls.addWidget(self.btn_back60)
        controls.addWidget(self.slider)
        controls.addWidget(self.btn_forward60)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.video_label)
        layout.addLayout(controls)
        self.setCentralWidget(container)
        
        # Initial render
        self.render_frame(0)
        
    def precompute_tracking(self):
        """Precompute all tracking data to eliminate runtime lag"""
        tracker = MotionDetector(activity=5,
                                threshold=0.05,
                                dis=30,
                                fskip=1,
                                N=10,
                                kf_params={'dt':1.0, 'accel_var':1.0, 'meas_var':1.0})
        
        # Create progress dialog
        progress = QProgressDialog("Precomputing tracking data...", 
                                  "Cancel", 0, self.total_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        tracking_data = []
        for i in range(self.total_frames):
            if progress.wasCanceled():
                break
            frame = self.video[i]
            objs = tracker.update(frame)
            tracking_data.append(objs)
            progress.setValue(i + 1)
            QApplication.processEvents()  # Keep UI responsive
        
        progress.close()
        return tracking_data
    
    def on_slider_change(self, value):
        self.render_frame(value)
    
    def jump_frames(self, offset):
        new_idx = np.clip(self.slider.value() + offset, 0, self.total_frames - 1)
        self.slider.setValue(int(new_idx))
    
    def render_frame(self, frame_idx):
        # Get precomputed tracking data
        objs = self.precomputed_tracking[frame_idx]
        
        # Grab and display frame
        frame = self.video[frame_idx]
        h, w, ch = frame.shape
        img = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        qimg = QImage(img.data, w, h, 3*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pix = pix.scaled(self.video_label.size(), 
                               Qt.KeepAspectRatio, 
                               Qt.SmoothTransformation)
        
        # Draw with QPainter
        painter = QPainter(scaled_pix)
        pen = QPen(QColor('red'))
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Calculate scaling factors for drawing
        scale_x = scaled_pix.width() / pix.width()
        scale_y = scaled_pix.height() / pix.height()
        
        # Draw trails
        for obj in objs:
            history = obj['history']
            if history:
                # Scale points to current display size
                pts = [(int(p[1] * scale_x), int(p[0] * scale_y)) for p in history]
                for i in range(1, len(pts)):
                    painter.drawLine(pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1])
        
        # Draw current centroids
        for obj in objs:
            y, x = obj['centroid']
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            painter.drawEllipse(scaled_x-5, scaled_y-5, 10, 10)
        
        painter.end()
        self.video_label.setPixmap(scaled_pix)
        self.current_frame = frame_idx

if __name__ == '__main__':
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print("\nEnter a file name such as: GUI.py [filename.mp4]")
        sys.exit()

    win = GUI(path)
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec())