"""
Object and Face Centroid Tracker with IoU and Spatial Velocity Estimation.
Maintains persistent track IDs across frames and monitors presence duration and movement.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class TrackedFace:
    """Represents a continuous facial track over time."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[int, int]         # (cx, cy)
    first_seen_sec: float
    last_seen_sec: float
    total_visible_sec: float = 0.0
    frame_count: int = 1
    disappeared_frames: int = 0
    matched_student_id: Optional[str] = None
    match_confidence: float = 0.0
    matched_student_name: Optional[str] = None
    movement_energy: float = 0.0      # Instantaneous displacement in pixels
    total_movement_energy: float = 0.0
    face_encoding: Optional[List[float]] = None


class CentroidTracker:
    """
    Centroid and IoU face tracker for continuous video streams.
    """

    def __init__(
        self,
        max_disappeared_frames: int = 30,
        max_centroid_distance: float = 85.0,
        iou_threshold: float = 0.20,
    ):
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedFace] = {}
        self.max_disappeared_frames = max_disappeared_frames
        self.max_centroid_distance = max_centroid_distance
        self.iou_threshold = iou_threshold

    @staticmethod
    def _compute_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = bbox
        return int(x + w / 2.0), int(y + h / 2.0)

    @staticmethod
    def _compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        xa1, ya1, wa, ha = box_a
        xa2, ya2 = xa1 + wa, ya1 + ha

        xb1, yb1, wb, hb = box_b
        xb2, yb2 = xb1 + wb, yb1 + hb

        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = wa * ha
        area_b = wb * hb
        union_area = float(area_a + area_b - inter_area)

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def register(self, bbox: Tuple[int, int, int, int], timestamp_sec: float) -> TrackedFace:
        """Register a new tracked face."""
        centroid = self._compute_centroid(bbox)
        track = TrackedFace(
            track_id=self.next_track_id,
            bbox=bbox,
            centroid=centroid,
            first_seen_sec=timestamp_sec,
            last_seen_sec=timestamp_sec,
            total_visible_sec=0.0,
            frame_count=1,
            disappeared_frames=0,
        )
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        return track

    def deregister(self, track_id: int) -> None:
        """Remove a track that disappeared for too long."""
        if track_id in self.tracks:
            del self.tracks[track_id]

    def update(
        self,
        rects: List[Tuple[int, int, int, int]],
        timestamp_sec: float,
    ) -> List[TrackedFace]:
        """
        Update tracking state with new detections.
        """
        # If no detections in current frame
        if len(rects) == 0:
            for track_id in list(self.tracks.keys()):
                track = self.tracks[track_id]
                track.disappeared_frames += 1
                track.movement_energy = 0.0
                if track.disappeared_frames > self.max_disappeared_frames:
                    self.deregister(track_id)
            return list(self.tracks.values())

        # If no existing tracks, register all detections
        if len(self.tracks) == 0:
            for bbox in rects:
                self.register(bbox, timestamp_sec)
            return list(self.tracks.values())

        # Grab existing track IDs and centroids
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]

        # Compute centroids for new detections
        input_centroids = [self._compute_centroid(box) for box in rects]

        # Compute distance matrix
        dists = np.zeros((len(track_ids), len(rects)), dtype=np.float32)
        ious = np.zeros((len(track_ids), len(rects)), dtype=np.float32)

        for i, (tcx, tcy) in enumerate(track_centroids):
            for j, (icx, icy) in enumerate(input_centroids):
                d = np.sqrt((tcx - icx) ** 2 + (tcy - icy) ** 2)
                dists[i, j] = d
                ious[i, j] = self._compute_iou(self.tracks[track_ids[i]].bbox, rects[j])

        # Match tracks to detections using greedy matching
        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # Check threshold (centroid distance or reasonable IoU)
            if dists[row, col] > self.max_centroid_distance and ious[row, col] < self.iou_threshold:
                continue

            track_id = track_ids[row]
            track = self.tracks[track_id]
            new_bbox = rects[col]
            new_centroid = input_centroids[col]

            # Calculate displacement
            disp = float(np.sqrt(
                (new_centroid[0] - track.centroid[0]) ** 2 +
                (new_centroid[1] - track.centroid[1]) ** 2
            ))
            track.movement_energy = disp
            track.total_movement_energy += disp

            # Calculate time delta
            dt = max(0.0, timestamp_sec - track.last_seen_sec)
            if dt < 2.0:  # Sensible continuous observation window
                track.total_visible_sec += dt

            track.bbox = new_bbox
            track.centroid = new_centroid
            track.last_seen_sec = timestamp_sec
            track.frame_count += 1
            track.disappeared_frames = 0

            used_rows.add(row)
            used_cols.add(col)

        # Unmatched existing tracks
        unused_rows = set(range(len(track_ids))) - used_rows
        for row in unused_rows:
            track_id = track_ids[row]
            track = self.tracks[track_id]
            track.disappeared_frames += 1
            track.movement_energy = 0.0
            if track.disappeared_frames > self.max_disappeared_frames:
                self.deregister(track_id)

        # Unmatched new detections
        unused_cols = set(range(len(rects))) - used_cols
        for col in unused_cols:
            self.register(rects[col], timestamp_sec)

        return list(self.tracks.values())
