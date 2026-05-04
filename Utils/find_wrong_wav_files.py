import sys
import os
import glob
import json
import numpy as np
import soundfile as sf
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

# Configuration
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
STATE_FILE = os.path.join(os.path.dirname(__file__), 'state.json')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'cleanup_dataset.json')

class AudioReviewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Dataset Cleanup")
        self.resize(800, 600)
        
        self.files = []
        self.current_index = 0
        self.current_audio = None
        self.current_sr = None
        self.is_playing = False
        
        self.undo_stack = []  # Store up to 10 previous decisions: list of (index, log_entry)
        
        self.categories = ['yes', 'no', 'up', 'down', 'other']  # Default
        self.load_categories_from_config()
        
        self.init_ui()
        self.load_files_and_sort()
        self.load_state()
        
        if self.files:
            self.load_file(self.current_index)
        else:
            self.info_label.setText("No .wav files found in dataset/")

    def load_categories_from_config(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = json.load(f)
                if 'keywords' in data:
                    self.categories = data['keywords']

    def load_files_and_sort(self):
        self.info_label.setText("Scanning and calculating RMS for all files... Please wait.")
        QApplication.processEvents()
        
        search_pattern = os.path.join(DATASET_DIR, '**', '*.wav')
        wav_files = glob.glob(search_pattern, recursive=True)
        
        file_rms = []
        for f in wav_files:
            try:
                data, _ = sf.read(f)
                if len(data) == 0:
                    rms = 0
                else:
                    rms = np.sqrt(np.mean(data**2))
                # Store relative path
                rel_path = os.path.relpath(f, DATASET_DIR)
                rel_path = rel_path.replace('\\', '/')
                file_rms.append((rel_path, rms))
            except Exception as e:
                print(f"Failed to read {f}: {e}")
                
        # Sort by RMS ascending
        file_rms.sort(key=lambda x: x[1])
        self.files = [x[0] for x in file_rms]
        
    def load_state(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
                self.current_index = data.get('current_index', 0)
                
        # Make sure index is valid
        if self.current_index >= len(self.files):
            self.current_index = len(self.files) - 1
        if self.current_index < 0:
            self.current_index = 0

    def save_state(self):
        with open(STATE_FILE, 'w') as f:
            json.dump({'current_index': self.current_index}, f)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Info label
        self.info_label = QLabel("Loading...")
        self.info_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.info_label)
        
        self.details_label = QLabel("")
        self.details_label.setStyleSheet("font-size: 14px;")
        self.details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.details_label)
        
        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        main_layout.addWidget(self.plot_widget)
        
        # Audio controls
        audio_layout = QHBoxLayout()
        self.play_pause_btn = QPushButton("Pause")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        audio_layout.addWidget(self.play_pause_btn)
        
        audio_layout.addWidget(QLabel("Volume Boost:"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(1)
        self.volume_slider.setMaximum(100) # representing 1.0 to 10.0
        self.volume_slider.setValue(10)
        self.volume_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.volume_slider.setTickInterval(10)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        audio_layout.addWidget(self.volume_slider)
        
        self.volume_label = QLabel("1.0x")
        audio_layout.addWidget(self.volume_label)
        
        main_layout.addLayout(audio_layout)
        
        # Action controls
        action_layout = QHBoxLayout()
        self.correct_btn = QPushButton("CORRECT (Space)")
        self.correct_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 20px; font-weight: bold; padding: 20px;")
        self.correct_btn.clicked.connect(lambda: self.log_action("keep"))
        
        self.delete_btn = QPushButton("DELETE (Enter)")
        self.delete_btn.setStyleSheet("background-color: #F44336; color: white; font-size: 16px; font-weight: bold; padding: 15px;")
        self.delete_btn.clicked.connect(lambda: self.log_action("delete"))
        
        action_layout.addWidget(self.correct_btn, stretch=2)
        action_layout.addWidget(self.delete_btn, stretch=1)
        
        main_layout.addLayout(action_layout)
        
        # Categories
        cat_layout = QHBoxLayout()
        for cat in self.categories:
            btn = QPushButton(cat)
            btn.setStyleSheet("padding: 10px; font-size: 14px;")
            btn.clicked.connect(lambda checked, c=cat: self.log_action(c))
            cat_layout.addWidget(btn)
        main_layout.addLayout(cat_layout)
        
        # Set focus to capture keys
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def load_file(self, index):
        if not self.files:
            return
            
        self.stop_audio()
        
        rel_path = self.files[index]
        full_path = os.path.join(DATASET_DIR, rel_path)
        
        try:
            self.current_audio, self.current_sr = sf.read(full_path)
        except Exception as e:
            self.info_label.setText(f"Error loading {rel_path}: {e}")
            self.current_audio = np.zeros(100)
            self.current_sr = 16000
            
        # UI updates
        expected_cat = os.path.dirname(rel_path)
        filename = os.path.basename(rel_path)
        duration = len(self.current_audio) / self.current_sr if self.current_sr else 0
        
        self.info_label.setText(f"File {index+1}/{len(self.files)}: {filename}")
        self.details_label.setText(f"Category: {expected_cat} | Duration: {duration:.2f}s")
        
        # Plot
        self.plot_widget.clear()
        self.plot_widget.plot(self.current_audio, pen='y')
        
        # Reset slider and volume
        self.volume_slider.blockSignals(True)
        self.volume_slider.setValue(10) # 1.0x
        self.volume_slider.blockSignals(False)
        self.volume_label.setText("1.0x")
        
        self.play_audio()

    def play_audio(self):
        if self.current_audio is None:
            return
        
        vol_multiplier = self.volume_slider.value() / 10.0
        audio_to_play = self.current_audio * vol_multiplier
        
        # clip to prevent horrible distortion sounds taking down the system
        audio_to_play = np.clip(audio_to_play, -1.0, 1.0)
        
        sd.stop()
        sd.play(audio_to_play, self.current_sr, loop=True)
        self.is_playing = True
        self.play_pause_btn.setText("Pause")

    def stop_audio(self):
        sd.stop()
        self.is_playing = False
        self.play_pause_btn.setText("Resume")

    def toggle_playback(self):
        if self.is_playing:
            self.stop_audio()
        else:
            self.play_audio()

    def on_volume_changed(self):
        vol_multiplier = self.volume_slider.value() / 10.0
        self.volume_label.setText(f"{vol_multiplier:.1f}x")
        if self.is_playing:
            self.play_audio()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.log_action("keep")
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self.log_action("delete")
        elif event.key() == Qt.Key.Key_Right:
            self.next_file()
        elif event.key() == Qt.Key.Key_Left:
            self.undo()
            
    def log_action(self, action):
        if not self.files:
            return
            
        rel_path = self.files[self.current_index]
        log_entry = {"filepath": rel_path, "action": action}
        
        # Append to log
        logs = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r') as f:
                    logs = json.load(f)
            except:
                pass
                
        logs.append(log_entry)
        
        with open(LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=2)
            
        # Add to undo stack
        self.undo_stack.append((self.current_index, log_entry))
        if len(self.undo_stack) > 10:
            self.undo_stack.pop(0)
            
        self.next_file()

    def next_file(self):
        if self.current_index < len(self.files) - 1:
            self.current_index += 1
            self.save_state()
            self.load_file(self.current_index)
        else:
            self.stop_audio()
            self.info_label.setText("Finished all files!")
            
    def undo(self):
        if not self.undo_stack:
            # If no undo stack but we just want to go back
            if self.current_index > 0:
                self.current_index -= 1
                self.save_state()
                self.load_file(self.current_index)
            return
            
        prev_index, log_entry = self.undo_stack.pop()
        
        # Remove from log file
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'r') as f:
                    logs = json.load(f)
                
                # Try to remove the exact entry from the end
                if logs and logs[-1] == log_entry:
                    logs.pop()
                    with open(LOG_FILE, 'w') as f:
                        json.dump(logs, f, indent=2)
            except:
                pass
                
        self.current_index = prev_index
        self.save_state()
        self.load_file(self.current_index)

    def closeEvent(self, event):
        self.stop_audio()
        self.save_state()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioReviewer()
    window.show()
    sys.exit(app.exec())
