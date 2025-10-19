import numpy as np
import pandas as pd
from config import N_LABELS, label_to_index


class Grid:
    def __init__(self, x_lims, y_lims, cell_size):
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.cell_size = cell_size
        self.n_x_cells = int((x_lims[1] - x_lims[0]) / cell_size)
        self.n_y_cells = int((y_lims[1] - y_lims[0]) / cell_size)
        self.n_channels = 3  # mean_rcs, mean_vr, count
        self.grid = np.zeros(
            (self.n_x_cells, self.n_y_cells, self.n_channels), dtype=np.float32
        )
        self.out_grid = np.zeros(
            (self.n_x_cells, self.n_y_cells, N_LABELS), dtype=np.uint8
        )

    def fill_grid(self, cur_dets: pd.DataFrame, is_output=False):
        for _, det in cur_dets.iterrows():
            x_cc = det["x_cc"]
            y_cc = det["y_cc"]

            if (
                self.x_lims[0] <= x_cc < self.x_lims[1]
                and self.y_lims[0] <= y_cc < self.y_lims[1]
            ):
                x_idx, y_idx = self.get_cell_id(x_cc, y_cc)

                if is_output:
                    self.out_grid[x_idx, y_idx, label_to_index[det["label_id"]]] = 1
                else:
                    self.grid[x_idx, y_idx, 0] += det["rcs"]
                    self.grid[x_idx, y_idx, 1] += det["vr_compensated"]
                    self.grid[x_idx, y_idx, 2] += 1

        if not is_output:
            self.grid[:, :, 0] /= np.maximum(self.grid[:, :, 2], 1)
            self.grid[:, :, 1] /= np.maximum(self.grid[:, :, 2], 1)

    def clear_grid(self):
        self.grid.fill(0)

    def clear_out_grid(self):
        self.out_grid.fill(0)

    def get_cell_id(self, x_cc, y_cc):
        x_idx = np.clip(
            round((x_cc - self.x_lims[0]) / self.cell_size), 0, self.n_x_cells - 1
        )
        y_idx = np.clip(
            round((y_cc - self.y_lims[0]) / self.cell_size), 0, self.n_y_cells - 1
        )
        return x_idx, y_idx

    def get_active_cells_positions(self):
        x_inds, y_inds = np.where(self.grid[:, :, 2] > 0)
        x_pos = x_inds * self.cell_size + self.x_lims[0]
        y_pos = y_inds * self.cell_size + self.y_lims[0]
        return x_inds, y_inds, x_pos, y_pos


if __name__ == "__main__":
    pass
    # Example usage
    # cur_dets = pd.DataFrame(
    #     {
    #         "x_cc": [5.2, 10.5, 15.3, 20.1, 25.4],
    #         "y_cc": [0.5, -10.2, 5.5, -15.3, 10.1],
    #         "rcs": [10, 20, 15, 25, 30],
    #         "vr_compensated": [1.5, -2.0, 0.5, -1.0, 2.5],
    #     }
    # )
    # grid_fl = Grid(x_lims=(2, 100), y_lims=(-50, 20), cell_size=0.5)
    # grid_fl.fill_grid(cur_dets)
