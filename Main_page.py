import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,QTabWidget, QWidget, QLabel, QPushButton, QFrame)
from PySide6.QtCore import Qt
import matplotlib
matplotlib.use("Qt5Agg")          # Ensure Qt backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt   # Still used for close/show if needed


class TabWidgetApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Crowd Monitoring Dashboard')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tab_widget = QTabWidget()
        self.Home_Page = QWidget()
        self.CSR = QWidget()
        self.IR = QWidget()

        self.tab_widget.addTab(self.Home_Page, "Home")
        self.tab_widget.addTab(self.CSR, "CSRNet")
        self.tab_widget.addTab(self.IR, "IR")
        self.layout.addWidget(self.tab_widget)

        self.warning_color = "green"  # Default warning state
        self.current_heatmap_index = 0

        # Single figure/axes for heatmap (no toolbar)
        self.heatmap_fig = None
        self.heatmap_ax = None
        self.heatmap_canvas = None

        self.initTab1()
        self.initTab2()
        self.initTab3()

    def initTab1(self):
        layout = QVBoxLayout(self.Home_Page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        bg_widget = QWidget(self.Home_Page)
        bg_widget.setObjectName("bgWidget")
        bg_layout = QVBoxLayout(bg_widget)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        bg_layout.setSpacing(0)

        label = QLabel("can you see me")
        label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        label.setStyleSheet("""
            color: white;
            background: transparent;
            font-size: 24px;
            font-weight: bold;
        """)
        label.setFixedHeight(40)
        bg_layout.addWidget(label)

        layout.addWidget(bg_widget)

        bg_widget.setStyleSheet(r"""
            #bgWidget {
                border-image: url(C:/Users/sanje/COLLEGE/Projects/MAIN_EL_SEM_1/GUI_home_page_bg_img1.jpg)
                              0 0 0 0 stretch stretch;
            }
        """)

    def initTab2(self):
        layout = QVBoxLayout(self.CSR)

        title = QLabel("CSRNet Crowd Analysis")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        button_layout = QHBoxLayout()

        
        self.retrieve_btn = QPushButton("Refresh Crowd Count")
        self.retrieve_btn.clicked.connect(self.retrieve_crowd_count)
        button_layout.addWidget(self.retrieve_btn)


        self.heatmap_btn = QPushButton("Display Heatmap (One-shot)")
        self.heatmap_btn.clicked.connect(self.show_heatmap_once)
        button_layout.addWidget(self.heatmap_btn)

        self.refresh_heatmap_btn = QPushButton("Refresh Heatmap")
        self.refresh_heatmap_btn.clicked.connect(self.refresh_heatmap)
        button_layout.addWidget(self.refresh_heatmap_btn)

        self.warning_btn = QPushButton("Warning Status")
        self.warning_btn.clicked.connect(self.check_warning_condition)
        button_layout.addWidget(self.warning_btn)

        layout.addLayout(button_layout)

        self.warning_frame = QFrame()
        self.warning_frame.setFixedSize(100, 100)
        self.warning_frame.setFrameShape(QFrame.StyledPanel)
        self.update_warning_color()
        layout.addWidget(self.warning_frame, alignment=Qt.AlignCenter)

        self.result_label = QLabel("Click buttons to analyze crowd data")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 14px; color: #666;")
        layout.addWidget(self.result_label)

    def initTab3(self):
        layout = QVBoxLayout(self.IR)
        label = QLabel("Infrared (IR) Analysis")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 16px;")
        layout.addWidget(label)
        layout.addStretch()

    # ---------------- Warning logic ----------------

    def check_warning_condition(self):
        colors = ["green", "orange", "red"]
        current_idx = colors.index(self.warning_color)
        next_color = colors[(current_idx + 1) % 3]
        self.set_warning_color(next_color)
        print(f"Warning updated to: {self.warning_color}")

    def set_warning_color(self, color):
        self.warning_color = color
        self.update_warning_color()

    def update_warning_color(self):
        color_map = {"green": "#00FF00", "orange": "#FFA500", "red": "#FF0000"}
        self.warning_frame.setStyleSheet(
            f"background-color: {color_map[self.warning_color]}; "
            "border: 3px solid black; border-radius: 50px;"
        )

    def retrieve_crowd_count(self):
        """
        Acts as a refresh: every click pulls the latest crowd count
        from the live CSRNet pipeline.
        """
        latest_count = self.get_latest_crowd_count()
        self.result_label.setText(f"Crowd count: {latest_count} people (Latest)")
        self.result_label.setStyleSheet("font-size: 14px; color: blue; font-weight: bold;")

    # ---------------- Heatmap logic ----------------

    def get_heatmap_data(self):
        """
        Replace this demo with your CSRNet live data:
            from csrnet_live import get_latest_heatmap_data
            return get_latest_heatmap_data()
        """
        demo_maps = [
            np.random.rand(20, 20),
            np.random.rand(10, 30),
            np.random.rand(30, 10)
        ]
        data = demo_maps[self.current_heatmap_index]
        self.current_heatmap_index = (self.current_heatmap_index + 1) % len(demo_maps)
        return data
    
    def get_latest_crowd_count(self):
        """
        Replace this demo with your real CSRNet live count.
        For now it just returns a random integer.
        """
        # from csrnet_live import get_latest_crowd_count
        # return get_latest_crowd_count()

        import random
        return random.randint(0, 200)  # DEMO


    def _ensure_heatmap_figure(self, title):
        """
        Create/reuse a clean figure WITHOUT toolbar.
        """
        if self.heatmap_fig is None or self.heatmap_ax is None:
            # Use Figure directly (no plt.figure toolbar)
            self.heatmap_fig = Figure(figsize=(8, 6))
            self.heatmap_ax = self.heatmap_fig.add_subplot(111)

            # Create a barebones window for this figure
            self.heatmap_canvas = FigureCanvasQTAgg(self.heatmap_fig)
            # Show in its own window
            self.heatmap_canvas.show()
        self.heatmap_ax.clear()
        self.heatmap_ax.set_title(title)

    def _plot_heatmap(self, data, title):
        """Create a fresh figure window for each plot and close the old one."""
        # Close old window if it exists
        if self.heatmap_canvas is not None:
            self.heatmap_canvas.close()
            self.heatmap_canvas = None
            self.heatmap_fig = None
            self.heatmap_ax = None

        # New clean figure (no toolbar)
        self.heatmap_fig = Figure(figsize=(8, 6))
        self.heatmap_ax = self.heatmap_fig.add_subplot(111)
        self.heatmap_ax.set_title(title)

        im = self.heatmap_ax.imshow(data, cmap='hot', aspect='auto')
        self.heatmap_fig.colorbar(im, ax=self.heatmap_ax)
        self.heatmap_fig.tight_layout()

        # Show in its own window
        self.heatmap_canvas = FigureCanvasQTAgg(self.heatmap_fig)
        self.heatmap_canvas.show()


    

    def show_heatmap_once(self):
        data = self.get_heatmap_data()
        self._plot_heatmap(data, "CSRNet Heatmap (Single-shot)")

    def refresh_heatmap(self):
        data = self.get_heatmap_data()
        self._plot_heatmap(data, "CSRNet Heatmap (Refreshed)")
        self.result_label.setText("Heatmap refreshed with latest data.")

        # If you ever want to fully close the heatmap window:
        # self.heatmap_canvas.close()
        # self.heatmap_fig = None
        # self.heatmap_ax = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tabWidgetApp = TabWidgetApp()
    tabWidgetApp.show()
    sys.exit(app.exec())
