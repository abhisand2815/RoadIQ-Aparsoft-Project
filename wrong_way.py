from dataclasses import dataclass, field
from typing import Dict, Tuple
import math

Point = Tuple[float, float]


@dataclass
class TrackState:
    positions: list[Point] = field(default_factory=list)
    wrong_frames: int = 0
    confirmed: bool = False


class WrongWayDetector:
    def __init__(
        self,
        traffic_direction="right",
        history_size=10,
        min_movement=10.0,
        confirmation_frames=5,
    ):
        self.traffic_direction = traffic_direction
        self.history_size = history_size
        self.min_movement = min_movement
        self.confirmation_frames = confirmation_frames
        self.tracks: Dict[int, TrackState] = {}

    def update(self, track_id: int, center: Point) -> bool:
        state = self.tracks.setdefault(track_id, TrackState())

        state.positions.append(center)

        if len(state.positions) > self.history_size:
            state.positions.pop(0)

        if len(state.positions) < 2:
            return False

        old_x, old_y = state.positions[0]
        new_x, new_y = state.positions[-1]

        dx = new_x - old_x
        dy = new_y - old_y

        distance = math.hypot(dx, dy)

        if distance < self.min_movement:
            return state.confirmed

        # Only consider predominantly horizontal movement.
        if abs(dx) < abs(dy):
            return state.confirmed

        if self.traffic_direction == "right":
            moving_wrong = dx < 0
        else:
            moving_wrong = dx > 0

        if moving_wrong:
            state.wrong_frames += 1
        else:
            state.wrong_frames = max(0, state.wrong_frames - 1)

        if state.wrong_frames >= self.confirmation_frames:
            state.confirmed = True

        return state.confirmed

    def remove_track(self, track_id: int):
        self.tracks.pop(track_id, None)

    def reset(self):
        self.tracks.clear()