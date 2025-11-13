from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QInputDialog
from PyQt6.QtCore import QTimer
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from classes import Map
from initialize_map import init_map, reset_to_init_pos
from velocity_control import append_vel_pos
from plots import animate_map, save_anim

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
        self.is_playing = False
        reset_btn = QPushButton("Reset")
        reinit_btn = QPushButton("Reinitialize")
        save_btn = QPushButton("Save")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(start_pause_btn)
        layout.addWidget(reset_btn)
        layout.addWidget(reinit_btn)
        layout.addWidget(save_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connect actions
        start_pause_btn.clicked.connect(self.start_pause_sim)
        reset_btn.clicked.connect(self.reset_sim)
        reinit_btn.clicked.connect(self.reinitialize_sim)
        save_btn.clicked.connect(self.save)

        # Timer for updates
        self.map_initialized = False
        self.map_timer = QTimer()
        self.map_timer.timeout.connect(self.update_map)
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_map_plot)

    def start_pause_sim(self):
        if self.is_playing:
            self.is_playing = False
            self.map_timer.stop()
            self.plot_timer.stop()
        else:
            self.is_playing = True
            self.map_timer.start(int(self.map.dt*1000))  # update every dt ms (100 Hz by default)
            self.plot_timer.start(30)                # update every 30 ms (~33 fps)

    def reset_sim(self):
        self.is_playing, self.map_initialized = False, False
        self.map_timer.stop()
        self.plot_timer.stop()
        reset_to_init_pos(self.map)
        
    def reinitialize_sim(self):
        self.is_playing, self.map_initialized = False, False
        self.map_timer.stop()
        self.plot_timer.stop()
        self.map = self.create_map_via_dialog()
        init_map(self.map)

    def update_map(self):
        append_vel_pos(self.map)

    def update_map_plot(self):
        if not self.map_initialized:
            self.initialize_map_plot()
            self.canvas.draw_idle()
        for i, a in enumerate(self.map.all_agents):
            x, y = a.p[0][-1], a.p[1][-1]
            self.agent_circles[i].center = (x, y)
        if not self.map.C_O_M.size:
            return
        for i, c in enumerate(self.map.C_O_M[-1]):
            x, y = c
            self.center_circles[i].center = (x, y)
        self.canvas.draw_idle()

    def initialize_map_plot(self):
        self.ax.clear()
        self.map_initialized = True

        self.agent_circles = []
        for a in self.map.all_agents:
            circle = patches.Circle(
                (a.p[0][0], a.p[1][0]),
                radius=a.r,
                color='blue',
                alpha=0.6)
            self.ax.add_patch(circle)
            self.agent_circles.append(circle)

        for o in self.map.all_obstacles:
            rect = patches.Rectangle(
                (o.x_min, o.y_min), # bottom left
                o.x_max - o.x_min,  # width
                o.y_max - o.y_min,  # height
                linewidth=1.5,
                edgecolor='red',
                facecolor='none'    # transparent inside
            )
            self.ax.add_patch(rect)

        self.center_circles = []
        for c in self.map.C_O_M[0]:
            circle = patches.Circle(
                    (c[0], c[1]), 
                    radius=2*self.map.all_agents[0].r, 
                    color='green', 
                    alpha=0.6)
            self.center_circles.append(circle)
            self.ax.add_patch(circle)

        self.plot_x_min, self.plot_x_max = self.map.x_min - 0.25*self.map.len_x, self.map.x_max + 0.25*self.map.len_x
        self.plot_y_min, self.plot_y_max = self.map.y_min - 0.25*self.map.len_y, self.map.y_max + 0.25*self.map.len_y
        self.ax.set_xlim(self.plot_x_min, self.plot_x_max)
        self.ax.set_ylim(self.plot_y_min, self.plot_y_max)

        self.ax.set_aspect('equal')
        self.ax.grid(True)

    def save(self):
        self.map_timer.stop()
        self.plot_timer.stop()
        print("Saving animation...")
        rect = [self.plot_x_min, self.plot_y_min, self.plot_x_max, self.plot_y_max]
        save_anim(animate_map(self.map,rectangle=rect))

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MapWindow()
    window.show()
    sys.exit(app.exec())