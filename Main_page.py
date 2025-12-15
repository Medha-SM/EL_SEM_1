import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, QPushButton, QFrame)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QPalette, QColor
import matplotlib.pyplot as plt

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
        self.initTab1()
        self.initTab2()
        self.initTab3()

    def initTab1(self):
        # Main layout for the tab
        layout = QVBoxLayout(self.Home_Page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Background widget that will fill the tab
        bg_widget = QWidget(self.Home_Page)
        bg_widget.setObjectName("bgWidget")  # important
        bg_layout = QVBoxLayout(bg_widget)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        bg_layout.setSpacing(0)

        # Label on top of background
        label = QLabel("CROWDS\nA CHALLENGE IN THE MAKING")
        label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        label.setStyleSheet("""
            color: white;
            background: transparent;
            font-size: 30px;
            font-weight: bold;
        """)
        bg_layout.addWidget(label)

        # Add background widget to tab layout
        layout.addWidget(bg_widget)

        # Style ONLY the bg_widget, not every QWidget
        bg_widget.setStyleSheet(r"""
            #bgWidget {
                border-image: url(C:/Users/sanje/COLLEGE/Projects/MAIN_EL_SEM_1/GUI_home_page_bg_img1.jpg)
                            0 0 0 0 stretch stretch;
            }
        """)

    


    def initTab2(self):
        layout = QVBoxLayout(self.CSR)
        
        # Title
        title = QLabel("CSRNet Crowd Analysis")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Buttons layout
        button_layout = QHBoxLayout()
        
        self.retrieve_btn = QPushButton("Retrieve Crowd Count")
        self.retrieve_btn.clicked.connect(self.retrieve_crowd_count)
        button_layout.addWidget(self.retrieve_btn)
        
        self.heatmap_btn = QPushButton("Display Heatmap")
        self.heatmap_btn.clicked.connect(self.show_heatmap)
        button_layout.addWidget(self.heatmap_btn)
        
        # Single warning button
        self.warning_btn = QPushButton("Warning Status")
        self.warning_btn.clicked.connect(self.check_warning_condition)
        button_layout.addWidget(self.warning_btn)
        
        layout.addLayout(button_layout)

        # Warning indicator
        self.warning_frame = QFrame()
        self.warning_frame.setFixedSize(100, 100)
        self.warning_frame.setFrameShape(QFrame.StyledPanel)
        self.update_warning_color()
        layout.addWidget(self.warning_frame, alignment=Qt.AlignCenter)

        # Placeholder for results
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

    def check_warning_condition(self):
        """Check condition and update warning color"""
        # TODO: Replace with actual condition
        # if [your_condition_here]:
        #     self.set_warning_color("red")
        # elif [another_condition]:
        #     self.set_warning_color("orange")
        # else:
        #     self.set_warning_color("green")
        
        # Demo: Cycle through colors for testing
        colors = ["green", "orange", "red"]
        current_idx = colors.index(self.warning_color)
        next_color = colors[(current_idx + 1) % 3]
        self.set_warning_color(next_color)
        
        print(f"Warning updated to: {self.warning_color}")

    def set_warning_color(self, color):
        """Control warning indicator color"""
        self.warning_color = color
        self.update_warning_color()

    def update_warning_color(self):
        """Update warning frame color"""
        color_map = {"green": "#00FF00", "orange": "#FFA500", "red": "#FF0000"}
        self.warning_frame.setStyleSheet(f"background-color: {color_map[self.warning_color]}; border: 3px solid black; border-radius: 50px;")

    def retrieve_crowd_count(self):
        """Placeholder for crowd count retrieval"""
        self.result_label.setText("Crowd count retrieved: 45 people (Demo)")
        self.result_label.setStyleSheet("font-size: 14px; color: blue; font-weight: bold;")

    def show_heatmap(self):
        """Provision for heatmap display using standalone matplotlib window"""
        plt.figure(figsize=(10, 8))
        im = plt.imshow([[1,2,3],[4,5,6],[7,8,9]], cmap='hot', aspect='auto')
        plt.colorbar(im)
        plt.title("Crowd Heatmap (Demo)")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tabWidgetApp = TabWidgetApp()
    tabWidgetApp.show()
    sys.exit(app.exec())
