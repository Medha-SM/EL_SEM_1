import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QTabWidget, QWidget, QLabel, QPushButton, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage

from model import CSRNet


##########################
# CONFIG
##########################
VIDEO_PATH = r"C:\Users\sanje\Downloads\WhatsApp Video 2025-12-16 at 9.52.59 AM.mp4"
WEIGHTS    = r"C:\Users\sanje\COLLEGE\Projects\MAIN_EL_SEM_1\weights.pth"
FRAME_STEP = 15
##########################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


##########################
# BACKEND
##########################
class CSRBackend:
    def __init__(self, video_path, weight_path, frame_step=15):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video")

        self.frame_step = frame_step
        self.model = CSRNet().to(DEVICE)
        self.model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        self.model.eval()

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = 0
        self.baseline_density = None
        self._prepare_baseline()

    def _prepare_baseline(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Empty video")
        self.baseline_density = self.get_density_map(frame)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_density_map(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            density = self.model(x)
        return density.squeeze().cpu().numpy()

    def adjust_density(self, density):
        d = density - self.baseline_density
        np.maximum(d, 0, out=d)
        if d.max() > 1e-6:
            d /= d.max()
        return d

    def _read_next_step_frame(self):
        for _ in range(self.frame_step - 1):
            if not self.cap.grab():
                return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def next_frame(self):
        frame = self._read_next_step_frame()
        if frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = self._read_next_step_frame()
            if frame is None:
                return None, None, None

        density = self.get_density_map(frame)
        density_norm = self.adjust_density(density)
        count = float(density_norm.sum())
        return frame, density_norm, count


##########################
# GUI
##########################
class CrowdDemoGUI(QMainWindow):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.setWindowTitle("CSRNet Crowd Monitoring Dashboard")
        self.setGeometry(100, 100, 1300, 700)

        self.tabs = QTabWidget()
        self.home = QWidget()
        self.csr = QWidget()

        self.tabs.addTab(self.home, "Home")
        self.tabs.addTab(self.csr, "CSRNet")

        self.setCentralWidget(self.tabs)
        self.init_home()
        self.init_csr()

    def init_home(self):
        layout = QVBoxLayout(self.home)
        label = QLabel("CSRNet Crowd Monitoring System")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size:26px; font-weight:bold;")
        layout.addWidget(label)

    def create_card(self):
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 14px;
                border: 1px solid #dddddd;
            }
        """)
        return card

    def init_csr(self):
        main = QVBoxLayout(self.csr)
        grid = QHBoxLayout()

        left = QVBoxLayout()
        right = QVBoxLayout()

        # Top-left: Frame
        frame_card = self.create_card()
        fl = QVBoxLayout(frame_card)
        fl.addWidget(QLabel("Original Frame", alignment=Qt.AlignCenter))
        self.frame_label = QLabel(alignment=Qt.AlignCenter)
        fl.addWidget(self.frame_label)
        left.addWidget(frame_card)

        # Bottom-left: Heatmap
        heat_card = self.create_card()
        hl = QVBoxLayout(heat_card)
        hl.addWidget(QLabel("Density Heatmap", alignment=Qt.AlignCenter))
        self.heatmap_label = QLabel(alignment=Qt.AlignCenter)
        hl.addWidget(self.heatmap_label)
        left.addWidget(heat_card)

        # Top-right: Status
        status_card = self.create_card()
        sl = QVBoxLayout(status_card)
        self.status_label = QLabel("NORMAL")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            background-color: green;
            color: white;
            font-size: 22px;
            font-weight: bold;
            padding: 25px;
            border-radius: 10px;
        """)
        sl.addWidget(self.status_label)
        right.addWidget(status_card)

        # Bottom-right: Stats
        stats_card = self.create_card()
        st = QVBoxLayout(stats_card)
        self.count_label = QLabel("Crowd Count : 0", alignment=Qt.AlignCenter)
        self.density_label = QLabel("Density / (4px × 4px) : 0", alignment=Qt.AlignCenter)
        for l in (self.count_label, self.density_label):
            l.setStyleSheet("font-size:16px; font-weight:bold;")
            st.addWidget(l)
        right.addWidget(stats_card)

        grid.addLayout(left, 2)
        grid.addLayout(right, 1)

        main.addLayout(grid)

        self.next_btn = QPushButton("Next Frame")
        self.next_btn.setFixedWidth(180)
        self.next_btn.clicked.connect(self.update_gui)
        main.addWidget(self.next_btn, alignment=Qt.AlignCenter)

    def frame_to_pixmap(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))

    def heatmap_to_pixmap(self, heatmap):
        heat = (heatmap * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        h, w, ch = heat.shape
        return QPixmap.fromImage(QImage(heat.data, w, h, ch * w, QImage.Format_RGB888))

    def update_gui(self):
        frame, heatmap, count = self.backend.next_frame()
        if frame is None:
            return

        # Display original frame
        self.frame_label.setPixmap(
            self.frame_to_pixmap(frame).scaled(450, 300, Qt.KeepAspectRatio)
        )

        # Display heatmap
        self.heatmap_label.setPixmap(
            self.heatmap_to_pixmap(heatmap).scaled(450, 300, Qt.KeepAspectRatio)
        )

        # Compute density per 4x4 pixel block
        h, w = heatmap.shape
        blocks = (h * w) / (4 * 4)
        density = count / blocks if blocks > 0 else 0

        # Update stats text
        self.count_label.setText(f"Crowd Count : {count:.2f}")
        self.density_label.setText(
            f"Density / (4px × 4px) : {density:.6f}"
        )

        # ───────── ALERT LOGIC ─────────
        if density < 0.2:
            status_text = "LOW"
            color = "green"
        elif density < 0.3:
            status_text = "MEDIUM"
            color = "yellow"
        else:
            status_text = "HIGH"
            color = "red"

        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"""
            background-color: {color};
            color: {'black' if color == 'yellow' else 'white'};
            font-size: 22px;
            font-weight: bold;
            padding: 25px;
            border-radius: 10px;
        """)


##########################
# MAIN
##########################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    backend = CSRBackend(VIDEO_PATH, WEIGHTS, FRAME_STEP)
    gui = CrowdDemoGUI(backend)
    gui.show()
    sys.exit(app.exec())
