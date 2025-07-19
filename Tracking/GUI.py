import sys
import argparse
import numpy as np
import skvideo.io
# NOTE to Self: video does not open if not included
import numpy
numpy.float = numpy.float64
numpy.int = numpy.int_

from PySide6 import QtCore, QtWidgets, QtGui

from MotionDetector import MotionDetector
from KalmanFilter import KalmanFilter



class QtTracker(QtWidgets.QWidget):
    def __init__(self, frames):
        super().__init__()
        self.frames = frames
        self.current_frame_idx = 0
        self.tracking_history = []
        
        # Tracking components
        self.motion_detector = MotionDetector(
            activity=5,
            threshold=0.05,
            dis=50,
            fskip=0,
            N=10
        )
        self.tracks = {}  # track_id: {'filter': KalmanFilter, 'history': []}
        self.next_track_id = 0
        
        # Create buttons
        self.btn_jump_forward = QtWidgets.QPushButton(">> 60 Frames")
        self.btn_jump_back = QtWidgets.QPushButton("<< 60 Frames")
        
        # Configure image label
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.update_image()
        
        # Configure slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(frames) - 1)
        self.frame_slider.setValue(0)
        
        # Create layout
        jump_layout = QtWidgets.QHBoxLayout()
        jump_layout.addWidget(self.btn_jump_back)
        jump_layout.addWidget(self.btn_jump_forward)
        
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.img_label)
        main_layout.addLayout(jump_layout)
        main_layout.addWidget(self.frame_slider)
        
        # Connect signals
        self.btn_jump_forward.clicked.connect(self.jump_forward)
        self.btn_jump_back.clicked.connect(self.jump_back)
        self.frame_slider.sliderMoved.connect(self.slider_moved)
        
    def jump_forward(self):
        self.jump_frames(60)
        
    def jump_back(self):
        self.jump_frames(-60)
        
    def jump_frames(self, delta):
        new_idx = max(0, min(len(self.frames) - 1, self.current_frame_idx + delta))
        self.current_frame_idx = new_idx
        self.frame_slider.setValue(new_idx)
        self.update_image()
        
    def slider_moved(self, position):
        self.current_frame_idx = position
        self.update_image()
        
    def update_image(self):
        """Process and display the current frame with tracking"""
        self.process_frame()
        
        frame = self.frames[self.current_frame_idx]
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        # Convert frame to uint8 if needed
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
            
        # Create QImage
        q_img = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        
        # Draw tracking information
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Draw detection history trails in red
        for track_id, track in self.tracks.items():
            # Draw red trail for history
            history = track['history']
            if len(history) > 1:
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))  # Red pen
                for i in range(1, len(history)):
                    painter.drawLine(
                        int(history[i-1][0]), int(history[i-1][1]),
                        int(history[i][0]), int(history[i][1])
                    )
            
            # Draw current position with ID
            if history:
                current_pos = history[-1]
                # Draw red circle
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))  # Red border
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 100)))  # Semi-transparent red fill
                painter.drawEllipse(QtCore.QPointF(current_pos[0], current_pos[1]), 10, 10)
                
                # Draw ID text
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))  # White text
                painter.drawText(
                    int(current_pos[0]) + 15, 
                    int(current_pos[1]) + 5, 
                    f"ID: {track_id}"
                )
            
        painter.end()
        
        self.img_label.setPixmap(pixmap)
        
    def process_frame(self):
        """Process the current frame for motion tracking"""
        # Reset tracker when going back to previous frames
        if self.current_frame_idx < len(self.tracking_history):
            # Restore tracker state from history
            self.tracks = self.tracking_history[self.current_frame_idx]['tracks'].copy()
            return
            
        frame = self.frames[self.current_frame_idx]
        detected_objects = self.motion_detector.detect_objects(frame)
        
        # Predict existing tracks
        for track_id, track in list(self.tracks.items()):
            track['filter'].predict()
            
        # Associate detections to tracks
        matched_detections = set()
        matched_tracks = set()
        
        for det in detected_objects:
            min_dist = float('inf')
            best_track = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                    
                pred_pos = track['filter'].get_position()
                det_pos = det[:2]
                dist = np.linalg.norm(pred_pos - det_pos)
                
                if dist < self.motion_detector.dis and dist < min_dist:
                    min_dist = dist
                    best_track = track_id
                    
            if best_track is not None:
                self.tracks[best_track]['filter'].update(det[:2])
                self.tracks[best_track]['history'].append(det[:2])
                matched_detections.add(tuple(det))
                matched_tracks.add(best_track)
                
        # Create new tracks for unmatched detections
        for det in detected_objects:
            if (tuple(det) not in matched_detections and 
                len(self.tracks) < self.motion_detector.N):
                initial_state = np.array([det[0], det[1], 0, 0])  # [x, y, vx, vy]
                self.tracks[self.next_track_id] = {
                    'filter': KalmanFilter(initial_state),
                    'history': [det[:2]]
                }
                self.next_track_id += 1
                
        # Remove stale tracks
        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_tracks:
                if len(track['history']) > self.motion_detector.activity:
                    del self.tracks[track_id]
                    
        # Store current state
        self.tracking_history.append({
            'frame_idx': self.current_frame_idx,
            'tracks': self.tracks.copy()
        })

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Motion Tracking with Kalman Filter")
    parser.add_argument("video_path", metavar='PATH_TO_VIDEO', type=str)
    parser.add_argument("--num_frames", metavar='n', type=int, default=-1)
    args = parser.parse_args()

    # Load video
    try:
        if args.num_frames > 0:
            frames = skvideo.io.vread(args.video_path, num_frames=args.num_frames)
        else:
            frames = skvideo.io.vread(args.video_path)
        print(f"Loaded video with shape: {frames.shape}")
    except Exception as e:
        print(f"Error loading video: {e}")
        sys.exit(1)

    # Create and run application
    app = QtWidgets.QApplication(sys.argv)
    
    widget = QtTracker(frames)
    widget.resize(1000, 700)
    widget.setWindowTitle("Motion Tracking GUI")
    widget.show()
    
    sys.exit(app.exec())