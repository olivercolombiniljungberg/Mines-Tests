from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QInputDialog
from PyQt6.QtCore import QTimer
import ast
import matplotlib.pyplot as plt
import numpy as np

from classes import Map
from initialize_map import init_map, reset_to_init_pos
from velocity_control import append_vel_pos

def parse_kwargs(s):
    """ Safely parse 'a=1, b=False' → {'a':1, 'b':False} """
    if not s.strip():
        return {}
    # Build a fake function call string to safely parse arguments
    try:
        tree = ast.parse(f"f({s})", mode="eval")
        kwargs = {}
        for kw in tree.body.keywords:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        return kwargs
    except Exception as e:
        print("❌ Invalid input:", e)
        return {}

class MapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.map = self.create_map_via_dialog()
        init_map(self.map)
        self.init_ui()

    def create_map_via_dialog(self):
        text, ok = QInputDialog.getText(
            self,
            "Map Initialization",
            "Enter Map() arguments:"
        )
        if ok:
            kwargs = parse_kwargs(text)
            print("✅ Using Map arguments:", kwargs)
            return Map(**kwargs)
        else:
            print("🟡 Using default Map()")
            return Map()

    def init_ui(self):
        self.setWindowTitle("Interactive Map")

        # Create matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.canvas = FigureCanvas(self.fig)
        self.ax.set_aspect('equal')

        # Buttons
        start_pause_btn = QPushButton("Start/Pause")
        self.is_started = False
        reset_btn = QPushButton("Reset")
        reinit_btn = QPushButton("Reinitialize")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(start_pause_btn)
        layout.addWidget(reset_btn)
        layout.addWidget(reinit_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connect actions
        start_pause_btn.clicked.connect(self.start_pause_sim)
        reset_btn.clicked.connect(self.reset_sim)
        reinit_btn.clicked.connect(self.reinitialize_sim)

        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_map)

    def start_pause_sim(self):
        if self.is_started:
            self.is_started = False
            self.timer.stop()
        else:
            self.is_started = True
            self.timer.start(30)  # update every 30 ms (~33 fps)

    def reset_sim(self):
        self.is_started = False
        self.timer.stop()
        reset_to_init_pos(self.map)
        
    def reinitialize_sim(self):
        self.is_started = False
        self.timer.stop()
        self.map = Map()
        init_map(self.map)

    def update_map(self):
        append_vel_pos(self.map)
        self.ax.clear()
        # ... draw agents, obstacles, targets ...
        self.canvas.draw()