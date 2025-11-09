import numpy as np
import pandas as pd


class LKF_CV:
    def __init__(self, X: np.ndarray, P: np.ndarray, qx: float, qv: float):
        self.X = X  # state vector
        self.P = P  # state covariance
        self.Q = np.diag([qx**2, qx**2, qv**2, qv**2])  # process noise covariance
        self.S_inv = np.linalg.inv(self.P[:2, :2])  # in the beginning, gross estimate

    def predict(self, dt: float):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray, R: np.ndarray):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y = z - H @ self.X  # innovation
        S = H @ self.P @ H.T + R  # innovation covariance
        self.S_inv = np.linalg.inv(S)
        K = self.P @ H.T @ self.S_inv  # Kalman gain
        self.X = self.X + K @ y
        self.P = (np.eye(len(self.X)) - K @ H) @ self.P


class PointTrack:
    def __init__(self, id_: int, x: float, y: float):
        X = np.array([x, y, 0.0, 0.0])
        P = np.diag([1.5**2, 1.5**2, 20**2, 20**2])
        self.id = id_
        self.filter = LKF_CV(
            X, P, qx=1.0, qv=7.0
        )  # high Qs because we not necessarily have CV model
        self.age = 1
        self.status = "tentative"
        self.hits = 0
        self.frames_since_update = 0

    @property
    def X(self) -> np.ndarray:
        return self.filter.X

    @property
    def P(self) -> np.ndarray:
        return self.filter.P

    @property
    def S_inv(self) -> np.ndarray:
        return self.filter.S_inv

    def predict(self, dt: float):
        self.filter.predict(dt)

    def update(self, meas_x: float, meas_y: float):
        z = np.array([meas_x, meas_y])
        R = np.diag(
            [1.5**2, 1.5**2]
        )  # not reallistic, measurement noise is acutally in polar coordiantes but range and bearing are not avialable
        self.filter.update(z, R)
        self.hits += 1
        self.frames_since_update = 0


class PointTrackList:
    def __init__(self):
        self.tracks: list[PointTrack] = []
        self.next_id = 0
        self.num_tracks = 0

    def add_track(self, x: float, y: float):
        track = PointTrack(self.next_id, x, y)
        self.tracks.append(track)
        self.next_id += 1
        self.num_tracks += 1

    def remove_track(self, track_id: int):
        self.tracks = [t for t in self.tracks if t.id != track_id]
        self.num_tracks = len(self.tracks)

    def to_dataframe(self, timestamp: int) -> pd.DataFrame:
        data = {
            "timestamp": [],
            "track_id": [],
            "x": [],
            "y": [],
            "vx": [],
            "vy": [],
            "age": [],
            "status": [],
        }
        for track in self.tracks:
            data["timestamp"].append(timestamp)
            data["track_id"].append(track.id)
            data["x"].append(track.X[0])
            data["y"].append(track.X[1])
            data["vx"].append(track.X[2])
            data["vy"].append(track.X[3])
            data["age"].append(track.age)
            data["status"].append(track.status)
        return pd.DataFrame(data)


def tracking_main(
    track_list: PointTrackList, detections: pd.DataFrame, timestamp: int, dt: float
) -> pd.DataFrame:
    # ego motion compensation, if not done velocities are relative to ego vehicle

    # Predict all tracks
    for track in track_list.tracks:
        track.predict(dt)

    # Data association (simple nearest neighbor within a gate)
    radius_gate = 6.0  # meters
    for track in track_list.tracks:
        idx_use_for_update = -1
        min_distance = float("inf")
        for det_idx, det_row in detections.iterrows():
            if detections.at[det_idx, "associated_track_id"] != -1:
                continue  # already associated
            det_pos = np.array([det_row["x_cc"], det_row["y_cc"]])
            track_pos = track.X[:2]
            distance = np.linalg.norm(det_pos - track_pos)
            if distance < radius_gate:
                # use Mahalanobis distance for association
                innov = det_pos - track_pos
                mahalanobis_distance = innov @ track.S_inv @ innov.T
                mahalanobis_threshold = (
                    7.815  # chi-square value for 2 DOF at 0.95 confidence
                )
                if mahalanobis_distance < mahalanobis_threshold:
                    detections.at[det_idx, "associated_track_id"] = track.id
                    if mahalanobis_distance < min_distance:
                        min_distance = mahalanobis_distance
                        idx_use_for_update = det_idx
        if idx_use_for_update != -1:
            detections.at[idx_use_for_update, "use_for_update"] = True

    # tracks update
    for track in track_list.tracks:
        associated_dets = detections[detections["associated_track_id"] == track.id]
        if associated_dets.empty:
            track.frames_since_update += 1
        else:
            for _, det_row in associated_dets.iterrows():
                if det_row["use_for_update"]:
                    track.update(det_row["x_cc"], det_row["y_cc"])

    # creating new tracks for unmatched detections
    unassociated_dets = detections[detections["associated_track_id"] == -1]
    for _, row in unassociated_dets.iterrows():
        track_list.add_track(row["x_cc"], row["y_cc"])

    # track maintenance
    for track in track_list.tracks:
        if track.frames_since_update > 3:
            track_list.remove_track(track.id)
        elif track.status == "tentative" and track.frames_since_update > 1:
            track_list.remove_track(track.id)
        else:
            track.age += 1
            if track.hits >= 3 and track.status == "tentative":
                track.status = "confirmed"

    tracks_df = track_list.to_dataframe(timestamp)
    return tracks_df
