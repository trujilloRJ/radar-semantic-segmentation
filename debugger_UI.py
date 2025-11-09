import json
import os
import sys
import numpy as np

from common import get_scene
from dataset import OutGridDataset
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QLabel,
    QErrorMessage,
    QPushButton,
    QHBoxLayout,
    QSizePolicy,
)
import pyqtgraph as pg
from PyQt6.QtCore import Qt  # QTimer no longer needed
from PyQt6.QtGui import QPixmap, QImage
from pyqtgraph.Qt import QtGui
from dotenv import load_dotenv
from encoder import Grid
from config import label_to_index, DONT_CARE

load_dotenv()

# Parameters
NUM_POINTS = 50
X_MIN, X_MAX = -70, 30
Y_MIN, Y_MAX = 0, 100
# ---------------------

# Loader parameters
EXP_NAME = "deep3_unet_b4_DoppFilt_WCE01_LN_ep3"
RESULTS_PATH = f"results/{EXP_NAME}"
GT_PATH = "data/validation/gt"
DETECTIONS_PATH = os.getenv("DATA_LOCATION")
SEQUENCE_ID = "sequence_101"
limit_sequences_to_gt = True  # set to True if only sequences with GT should be loaded
grid = Grid(x_lims=(2, 100), y_lims=(-50, 20), cell_size=0.5)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Semantic Segmentation Debugger")
        self.setGeometry(100, 100, 1000, 600)  # x, y, width, height

        self.central_widget = QWidget()
        self.right_widget = QWidget()
        self.graph_widget = pg.PlotWidget()
        view_box = self.graph_widget.getViewBox()
        view_box.invertX(True)
        self.frame_image = QLabel()
        self.sequence_label = QLabel()

        self.layout = QHBoxLayout(self.central_widget)
        self.layout.addWidget(self.graph_widget)
        self.layout.addWidget(self.right_widget)
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.addWidget(self.frame_image)
        self.right_layout.addWidget(self.sequence_label)

        self.frame_image.setFixedSize(400, 200)
        self.frame_image.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.frame_image.setScaledContents(True)

        self.setCentralWidget(self.central_widget)

        self.setup_xy()

        # data
        self.sequence_id = SEQUENCE_ID
        self.load_sequence()
        try:
            self.gt_dataset = OutGridDataset(GT_PATH)
            self.gt_available = True
        except Exception as e:
            print("No gt found")
            self.gt_available = False
        try:
            self.pred_dataset = OutGridDataset(RESULTS_PATH)
            self.pred_dataset_available = True
        except Exception as e:
            print("No predictions found")
            self.pred_dataset_available = False

        self.update()

    def load_sequence(self):
        scene_fn = os.path.join(
            os.getenv("DATA_LOCATION"), self.sequence_id, "scenes.json"
        )
        self.detections = get_scene(scene_fn)
        self.detections.loc[:, "timestamp"] = self.detections["timestamp"].astype(str)
        self.timestamps = self.detections["timestamp"].unique().tolist()
        self.n_frames = len(self.timestamps)
        self.current_index = 0

    def update(self):
        self.draw_detections()
        self.update_text_elements()
        self.update_frame_image()
        self.draw_grid_cells()

    def update_frame_image(self):
        ts = self.timestamps[self.current_index]

        with open(
            f"{os.getenv('DATA_LOCATION')}\\{self.sequence_id}\\scenes.json"
        ) as f:
            data = json.load(f)
            image_name = data["scenes"][ts]["image_name"]

        img_path = (
            f"{os.getenv('DATA_LOCATION')}\\{self.sequence_id}\\camera\\{image_name}"
        )
        self.frame_image.setPixmap(QPixmap(img_path))

    def update_text_elements(self):
        self.frame_text.setText(f"Frame: {self.current_index + 1}/{self.n_frames}")
        self.sequence_label.setText(f"Sequence: {self.sequence_id}")

    def draw_detections(self):
        cur_dets = self.detections[
            self.detections["timestamp"] == self.timestamps[self.current_index]
        ]
        colors = [
            pg.intColor(cur_dets["label_id"].iloc[i].item(), hues=len(label_to_index))
            for i in range(len(cur_dets))
        ]
        self._det_plot.setData(
            cur_dets["y_cc"].values, cur_dets["x_cc"].values, brush=colors
        )

    def draw_grid_cells(self):
        if self.sequence_id not in self.gt_dataset.sequences:
            self._gt_plot.clear()
            self._pred_plot.clear()
            return

        ts = self.timestamps[self.current_index]
        gt_cells = (
            self.gt_dataset.get_data_by_sequence_ts(self.sequence_id, ts)
            .squeeze(0)
            .numpy()
        )
        gt_mask = gt_cells < label_to_index[DONT_CARE]
        x_inds, y_inds = np.where(gt_mask)
        x_pos = x_inds * grid.cell_size + grid.x_lims[0]
        y_pos = y_inds * grid.cell_size + grid.y_lims[0]
        colors = [
            pg.intColor(gt_cells[x_inds[i], y_inds[i]], hues=len(label_to_index))
            for i in range(len(x_inds))
        ]
        self._gt_plot.setData(y_pos, x_pos, brush=colors)

        if self.pred_dataset_available:
            pred_cells = (
                self.pred_dataset.get_data_by_sequence_ts(self.sequence_id, ts)
                .squeeze(0)
                .numpy()
            )
            mask = (pred_cells < label_to_index[DONT_CARE]) & gt_mask
            x_inds, y_inds = np.where(mask)
            x_pos = x_inds * grid.cell_size + grid.x_lims[0]
            y_pos = y_inds * grid.cell_size + grid.y_lims[0]
            colors = [
                pg.intColor(pred_cells[x_inds[i], y_inds[i]], hues=len(label_to_index))
                for i in range(len(x_inds))
            ]
            self._pred_plot.setData(y_pos, x_pos, brush=colors)

    def keyPressEvent(self, event):
        key_steps = {
            Qt.Key.Key_D: 1,
            Qt.Key.Key_A: -1,
            Qt.Key.Key_W: 10,
            Qt.Key.Key_S: -10,
        }

        key = event.key()
        if key in key_steps:
            if getattr(self, "n_frames", 0) > 0:
                step = key_steps[key]
            # wrap-around using modulo
            self.current_index = (self.current_index + step) % self.n_frames
            self.update()
        elif key == Qt.Key.Key_Q:
            seq_number = int(self.sequence_id.split("_")[-1])
            seq_number -= 1
            self.sequence_id = f"sequence_{seq_number:d}"
            self.load_sequence()
            self.update()
        elif key == Qt.Key.Key_E:
            seq_number = int(self.sequence_id.split("_")[-1])
            seq_number += 1
            self.sequence_id = f"sequence_{seq_number:d}"
            self.load_sequence()
            self.update()
        else:
            super().keyPressEvent(event)

    def setup_xy(self):
        self.graph_widget.setXRange(X_MIN, X_MAX)
        self.graph_widget.setYRange(Y_MIN, Y_MAX)
        self.graph_widget.setLabel("left", "Y-Axis")
        self.graph_widget.setLabel("bottom", "X-Axis")
        self.graph_widget.showGrid(x=True, y=True, alpha=0.3)

        # frame text
        font = QtGui.QFont()

        # 3. Customize the font properties
        font.setFamily("Courier New")
        font.setPointSize(16)
        font.setWeight(500)
        self.frame_text = pg.TextItem(text="", anchor=(0.5, 0.5), color=(255, 255, 255))
        self.frame_text.setFont(font)
        self.frame_text.setPos(X_MAX - 15, Y_MAX)

        # gt grid
        self._gt_plot = pg.ScatterPlotItem(
            pen=pg.mkPen(width=0),
            size=10,
            symbol="s",
        )
        # pred grid
        self._pred_plot = pg.ScatterPlotItem(
            pen=pg.mkPen(width=0),
            size=6,
            symbol="o",
        )
        # detections
        self._det_plot = pg.ScatterPlotItem(
            pen=pg.mkPen(width=0),
            size=4,
            symbol="o",
        )

        self.graph_widget.addItem(self.frame_text)
        self.graph_widget.addItem(self._gt_plot)
        self.graph_widget.addItem(self._pred_plot)
        self.graph_widget.addItem(self._det_plot)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
