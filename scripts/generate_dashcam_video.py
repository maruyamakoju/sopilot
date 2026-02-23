"""
Generate realistic dashcam footage for insurance demo

Creates 3 types of videos:
1. Collision (rear-end accident)
2. Near-miss (sudden braking)
3. Normal driving

High-quality, professional-grade for demo purposes
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class DashcamVideoGenerator:
    """Generate realistic dashcam footage"""

    def __init__(self, width=1920, height=1080, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

    def generate_collision_video(self, output_path: str, duration_sec: int = 30):
        """
        Generate rear-end collision scenario

        Timeline:
        0-10s: Normal highway driving
        10-15s: Car ahead starts slowing
        15-20s: Brake lights visible, getting closer
        20-25s: COLLISION (impact frame)
        25-30s: Aftermath (stopped)
        """
        print(f"Generating collision video: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        total_frames = duration_sec * self.fps
        collision_frame = int(20 * self.fps)  # Collision at 20s

        for frame_idx in range(total_frames):
            img = self._create_highway_frame(frame_idx, total_frames, collision_frame)
            out.write(img)

            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}/{total_frames}")

        out.release()
        print(f"[OK] Collision video saved: {output_path}")

        # Add audio (engine sound + crash)
        self._add_audio(output_path, "collision")

        return {
            "type": "collision",
            "duration_sec": duration_sec,
            "collision_frame": collision_frame / self.fps,
            "severity": "HIGH",
            "ground_truth": {
                "fault_ratio": 100.0,  # Rear-ender is 100% at fault
                "scenario": "rear_end",
                "fraud_risk": 0.0,
            },
        }

    def generate_near_miss_video(self, output_path: str, duration_sec: int = 30):
        """
        Generate near-miss scenario

        Timeline:
        0-12s: Normal city driving
        12-15s: Pedestrian starts crossing
        15-18s: SUDDEN BRAKE (danger moment)
        18-22s: Slow down, pedestrian passes
        22-30s: Resume normal driving
        """
        print(f"Generating near-miss video: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        total_frames = duration_sec * self.fps
        near_miss_frame = int(15 * self.fps)  # Near-miss at 15s

        for frame_idx in range(total_frames):
            img = self._create_city_frame(frame_idx, total_frames, near_miss_frame)
            out.write(img)

            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}/{total_frames}")

        out.release()
        print(f"[OK] Near-miss video saved: {output_path}")

        self._add_audio(output_path, "near_miss")

        return {
            "type": "near_miss",
            "duration_sec": duration_sec,
            "near_miss_frame": near_miss_frame / self.fps,
            "severity": "MEDIUM",
            "ground_truth": {
                "fault_ratio": 0.0,  # No collision occurred
                "scenario": "pedestrian_avoidance",
                "fraud_risk": 0.0,
            },
        }

    def generate_normal_video(self, output_path: str, duration_sec: int = 30):
        """
        Generate normal driving scenario

        Timeline:
        0-30s: Normal highway driving, no incidents
        """
        print(f"Generating normal driving video: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        total_frames = duration_sec * self.fps

        for frame_idx in range(total_frames):
            img = self._create_normal_highway_frame(frame_idx, total_frames)
            out.write(img)

            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}/{total_frames}")

        out.release()
        print(f"[OK] Normal driving video saved: {output_path}")

        self._add_audio(output_path, "normal")

        return {
            "type": "normal",
            "duration_sec": duration_sec,
            "severity": "NONE",
            "ground_truth": {"fault_ratio": 0.0, "scenario": "normal_driving", "fraud_risk": 0.0},
        }

    def _create_highway_frame(self, frame_idx: int, total_frames: int, collision_frame: int):
        """Create realistic highway frame with approaching vehicle"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Sky (gradient)
        sky_color_top = (180, 120, 60)  # Light blue
        sky_color_bottom = (220, 180, 140)
        for y in range(self.height // 2):
            ratio = y / (self.height // 2)
            color = tuple(int(sky_color_top[i] * (1 - ratio) + sky_color_bottom[i] * ratio) for i in range(3))
            cv2.line(img, (0, y), (self.width, y), color, 1)

        # Road (gray)
        road_y = self.height // 2
        cv2.rectangle(img, (0, road_y), (self.width, self.height), (80, 80, 80), -1)

        # Lane markings (white dashed lines)
        lane_positions = [self.width // 3, 2 * self.width // 3]
        dash_length = 50
        dash_gap = 50
        offset = (frame_idx * 5) % (dash_length + dash_gap)

        for lane_x in lane_positions:
            y = road_y
            while y < self.height:
                y_start = y - offset
                y_end = min(y_start + dash_length, self.height)
                if y_start < self.height:
                    cv2.line(img, (lane_x, max(y_start, road_y)), (lane_x, y_end), (255, 255, 255), 8)
                y += dash_length + dash_gap

        # Car ahead (getting closer as we approach collision)
        progress = frame_idx / total_frames
        distance_factor = 1.0 - (frame_idx / collision_frame) if frame_idx < collision_frame else 0.1
        distance_factor = max(0.1, min(1.0, distance_factor))

        # Car size increases as it gets closer
        car_width = int(200 * (1.5 - distance_factor))
        car_height = int(150 * (1.5 - distance_factor))
        car_x = self.width // 2 - car_width // 2
        car_y_base = int(road_y + 100 + (self.height - road_y - 300) * distance_factor)
        car_y = car_y_base

        # Car body (dark blue)
        cv2.rectangle(img, (car_x, car_y), (car_x + car_width, car_y + car_height), (139, 69, 19), -1)
        cv2.rectangle(img, (car_x, car_y), (car_x + car_width, car_y + car_height), (200, 200, 200), 3)

        # Windows
        window_y = car_y + 20
        window_height = car_height // 3
        cv2.rectangle(
            img, (car_x + 30, window_y), (car_x + car_width - 30, window_y + window_height), (100, 150, 200), -1
        )

        # Brake lights (RED) if approaching collision
        if frame_idx > collision_frame - 150:  # 5 seconds before collision
            brake_intensity = min(255, int(255 * (collision_frame - frame_idx) / -150))
            brake_color = (0, 0, min(255, 150 + brake_intensity))
            tail_light_y = car_y + car_height - 30
            cv2.rectangle(img, (car_x + 10, tail_light_y), (car_x + 40, tail_light_y + 20), brake_color, -1)
            cv2.rectangle(
                img,
                (car_x + car_width - 40, tail_light_y),
                (car_x + car_width - 10, tail_light_y + 20),
                brake_color,
                -1,
            )

        # Collision effect (white flash + impact)
        if abs(frame_idx - collision_frame) < 5:
            flash_intensity = 255 - abs(frame_idx - collision_frame) * 50
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (255, 255, 255), -1)
            cv2.addWeighted(overlay, flash_intensity / 255.0, img, 1 - flash_intensity / 255.0, 0, img)

            # Impact text
            cv2.putText(
                img,
                "COLLISION!",
                (self.width // 2 - 200, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                8,
            )

        # Dashboard overlay (timestamp, speed)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        speed = int(60 - (frame_idx / collision_frame) * 40) if frame_idx < collision_frame else 0
        cv2.putText(img, timestamp, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img, f"Speed: {speed} km/h", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        return img

    def _create_city_frame(self, frame_idx: int, total_frames: int, near_miss_frame: int):
        """Create city driving frame with pedestrian near-miss"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Sky
        cv2.rectangle(img, (0, 0), (self.width, self.height // 2), (200, 180, 140), -1)

        # Road
        road_y = self.height // 2
        cv2.rectangle(img, (0, road_y), (self.width, self.height), (60, 60, 60), -1)

        # Crosswalk (white stripes)
        if abs(frame_idx - near_miss_frame) < 100:
            stripe_width = 50
            stripe_gap = 30
            crosswalk_y = road_y + 200
            for x in range(0, self.width, stripe_width + stripe_gap):
                cv2.rectangle(img, (x, crosswalk_y), (x + stripe_width, crosswalk_y + 150), (255, 255, 255), -1)

        # Pedestrian (if near near-miss moment)
        if abs(frame_idx - near_miss_frame) < 120:
            progress = (frame_idx - (near_miss_frame - 120)) / 240
            ped_x = int(100 + progress * (self.width - 200))
            ped_y = road_y + 220

            # Simple pedestrian figure
            cv2.circle(img, (ped_x, ped_y), 25, (0, 100, 200), -1)  # Head
            cv2.rectangle(img, (ped_x - 20, ped_y + 25), (ped_x + 20, ped_y + 80), (0, 100, 200), -1)  # Body
            cv2.line(img, (ped_x - 20, ped_y + 50), (ped_x - 50, ped_y + 80), (0, 100, 200), 8)  # Left arm
            cv2.line(img, (ped_x + 20, ped_y + 50), (ped_x + 50, ped_y + 80), (0, 100, 200), 8)  # Right arm
            cv2.line(img, (ped_x - 10, ped_y + 80), (ped_x - 30, ped_y + 130), (0, 100, 200), 8)  # Left leg
            cv2.line(img, (ped_x + 10, ped_y + 80), (ped_x + 30, ped_y + 130), (0, 100, 200), 8)  # Right leg

        # Warning indicator (if close to near-miss)
        if abs(frame_idx - near_miss_frame) < 30:
            cv2.putText(
                img, "! PEDESTRIAN !", (self.width // 2 - 250, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6
            )

        # Dashboard
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        speed = 40 if abs(frame_idx - near_miss_frame) > 60 else int(40 * (abs(frame_idx - near_miss_frame) / 60))
        cv2.putText(img, timestamp, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img, f"Speed: {speed} km/h", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        return img

    def _create_normal_highway_frame(self, frame_idx: int, total_frames: int):
        """Create normal highway driving frame"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Sky
        cv2.rectangle(img, (0, 0), (self.width, self.height // 2), (180, 140, 100), -1)

        # Road
        road_y = self.height // 2
        cv2.rectangle(img, (0, road_y), (self.width, self.height), (80, 80, 80), -1)

        # Lane markings (animated)
        lane_positions = [self.width // 3, 2 * self.width // 3]
        dash_length = 50
        dash_gap = 50
        offset = (frame_idx * 5) % (dash_length + dash_gap)

        for lane_x in lane_positions:
            y = road_y
            while y < self.height:
                y_start = y - offset
                y_end = min(y_start + dash_length, self.height)
                if y_start < self.height:
                    cv2.line(img, (lane_x, max(y_start, road_y)), (lane_x, y_end), (255, 255, 255), 8)
                y += dash_length + dash_gap

        # Distant car (far ahead, not a threat)
        car_x = self.width // 2 - 60
        car_y = road_y + 50
        cv2.rectangle(img, (car_x, car_y), (car_x + 120, car_y + 80), (100, 50, 50), -1)
        cv2.rectangle(img, (car_x, car_y), (car_x + 120, car_y + 80), (150, 150, 150), 2)

        # Dashboard
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, timestamp, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img, "Speed: 80 km/h", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        return img

    def generate_collision_intersection_video(self, output_path: str, duration_sec: int = 30):
        """Generate T-bone intersection collision scenario."""
        print(f"Generating intersection collision video: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        total_frames = duration_sec * self.fps
        collision_frame = int(18 * self.fps)

        for frame_idx in range(total_frames):
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Sky
            cv2.rectangle(img, (0, 0), (self.width, self.height // 2), (200, 180, 140), -1)
            # Road (intersection)
            road_y = self.height // 2
            cv2.rectangle(img, (0, road_y), (self.width, self.height), (70, 70, 70), -1)
            # Cross road
            cv2.rectangle(img, (self.width // 3, road_y), (2 * self.width // 3, self.height), (60, 60, 60), -1)

            # Approaching vehicle from the right
            if frame_idx > collision_frame - 90:
                progress = min(1.0, (frame_idx - (collision_frame - 90)) / 90)
                veh_x = int(self.width - 100 - progress * (self.width // 2 - 100))
                veh_y = road_y + 150
                cv2.rectangle(img, (veh_x, veh_y), (veh_x + 180, veh_y + 100), (0, 0, 180), -1)

            # Collision flash
            if abs(frame_idx - collision_frame) < 5:
                flash = 255 - abs(frame_idx - collision_frame) * 50
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (self.width, self.height), (255, 255, 255), -1)
                cv2.addWeighted(overlay, flash / 255.0, img, 1 - flash / 255.0, 0, img)
                cv2.putText(img, "COLLISION!", (self.width // 2 - 200, self.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)

            # Dashboard
            speed = 50 if frame_idx < collision_frame else 0
            cv2.putText(img, f"Speed: {speed} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            out.write(img)

        out.release()
        print(f"[OK] Intersection collision video saved: {output_path}")
        return {
            "type": "collision_intersection",
            "duration_sec": duration_sec,
            "severity": "HIGH",
            "ground_truth": {"fault_ratio": 0.0, "scenario": "t_bone", "fraud_risk": 0.0},
        }

    def generate_near_miss_cyclist_video(self, output_path: str, duration_sec: int = 30):
        """Generate near-miss with cyclist scenario."""
        print(f"Generating near-miss cyclist video: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        total_frames = duration_sec * self.fps
        event_frame = int(16 * self.fps)

        for frame_idx in range(total_frames):
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.rectangle(img, (0, 0), (self.width, self.height // 2), (200, 180, 140), -1)
            road_y = self.height // 2
            cv2.rectangle(img, (0, road_y), (self.width, self.height), (70, 70, 70), -1)

            # Cyclist crossing
            if abs(frame_idx - event_frame) < 90:
                progress = (frame_idx - (event_frame - 90)) / 180
                cx = int(self.width * 0.8 - progress * self.width * 0.6)
                cy = road_y + 180
                # Bicycle wheel circles + body
                cv2.circle(img, (cx, cy + 30), 25, (0, 200, 0), 3)
                cv2.circle(img, (cx + 50, cy + 30), 25, (0, 200, 0), 3)
                cv2.line(img, (cx, cy + 5), (cx + 50, cy + 5), (0, 200, 0), 3)
                cv2.circle(img, (cx + 25, cy - 25), 15, (0, 200, 0), -1)

            if abs(frame_idx - event_frame) < 30:
                cv2.putText(img, "! CYCLIST !", (self.width // 2 - 200, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)

            speed = 45 if abs(frame_idx - event_frame) > 60 else max(0, int(45 * abs(frame_idx - event_frame) / 60))
            cv2.putText(img, f"Speed: {speed} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            out.write(img)

        out.release()
        print(f"[OK] Near-miss cyclist video saved: {output_path}")
        return {
            "type": "near_miss_cyclist",
            "duration_sec": duration_sec,
            "severity": "MEDIUM",
            "ground_truth": {"fault_ratio": 0.0, "scenario": "cyclist_avoidance", "fraud_risk": 0.0},
        }

    def generate_near_miss_vehicle_video(self, output_path: str, duration_sec: int = 30):
        """Generate near-miss with vehicle cut-in scenario."""
        print(f"Generating near-miss vehicle video: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        total_frames = duration_sec * self.fps
        event_frame = int(14 * self.fps)

        for frame_idx in range(total_frames):
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.rectangle(img, (0, 0), (self.width, self.height // 2), (180, 140, 100), -1)
            road_y = self.height // 2
            cv2.rectangle(img, (0, road_y), (self.width, self.height), (80, 80, 80), -1)

            # Vehicle cutting in from the right lane
            if frame_idx > event_frame - 60:
                progress = min(1.0, (frame_idx - (event_frame - 60)) / 60)
                veh_x = int(self.width * 0.7 - progress * self.width * 0.2)
                veh_y = road_y + 100 + int(progress * 150)
                w = int(150 + progress * 100)
                h = int(100 + progress * 70)
                cv2.rectangle(img, (veh_x, veh_y), (veh_x + w, veh_y + h), (180, 50, 50), -1)
                cv2.rectangle(img, (veh_x, veh_y), (veh_x + w, veh_y + h), (200, 200, 200), 2)

            if abs(frame_idx - event_frame) < 20:
                cv2.putText(img, "! VEHICLE CUT-IN !", (self.width // 2 - 300, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

            speed = 70 if abs(frame_idx - event_frame) > 45 else max(10, int(70 * abs(frame_idx - event_frame) / 45))
            cv2.putText(img, f"Speed: {speed} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            out.write(img)

        out.release()
        print(f"[OK] Near-miss vehicle video saved: {output_path}")
        return {
            "type": "near_miss_vehicle",
            "duration_sec": duration_sec,
            "severity": "MEDIUM",
            "ground_truth": {"fault_ratio": 0.0, "scenario": "vehicle_cut_in", "fraud_risk": 0.0},
        }

    def generate_normal_city_video(self, output_path: str, duration_sec: int = 30):
        """Generate normal city driving scenario."""
        print(f"Generating normal city video: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        total_frames = duration_sec * self.fps

        for frame_idx in range(total_frames):
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.rectangle(img, (0, 0), (self.width, self.height // 2), (200, 180, 140), -1)
            road_y = self.height // 2
            cv2.rectangle(img, (0, road_y), (self.width, self.height), (70, 70, 70), -1)

            # Traffic light
            tl_x, tl_y = self.width // 2 + 200, road_y - 200
            cv2.rectangle(img, (tl_x, tl_y), (tl_x + 40, tl_y + 100), (50, 50, 50), -1)
            # Green light
            cv2.circle(img, (tl_x + 20, tl_y + 80), 12, (0, 255, 0), -1)

            # Buildings on sides
            cv2.rectangle(img, (50, road_y - 250), (250, road_y), (150, 130, 110), -1)
            cv2.rectangle(img, (self.width - 250, road_y - 200), (self.width - 50, road_y), (140, 120, 100), -1)

            cv2.putText(img, "Speed: 35 km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            out.write(img)

        out.release()
        print(f"[OK] Normal city video saved: {output_path}")
        return {
            "type": "normal_city",
            "duration_sec": duration_sec,
            "severity": "NONE",
            "ground_truth": {"fault_ratio": 0.0, "scenario": "normal_city", "fraud_risk": 0.0},
        }

    def generate_low_parking_bump_video(self, output_path: str, duration_sec: int = 30):
        """Generate low-speed parking lot bump scenario."""
        print(f"Generating parking bump video: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        total_frames = duration_sec * self.fps
        bump_frame = int(20 * self.fps)

        for frame_idx in range(total_frames):
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Parking lot surface
            cv2.rectangle(img, (0, 0), (self.width, self.height), (90, 90, 90), -1)
            # Parking lines
            for x in range(0, self.width, 200):
                cv2.line(img, (x, 200), (x, self.height - 200), (255, 255, 255), 3)

            # Parked car ahead (slowly approaching)
            car_y = 300 + max(0, int(200 * (1.0 - frame_idx / bump_frame))) if frame_idx < bump_frame else 300
            cv2.rectangle(img, (self.width // 2 - 100, car_y), (self.width // 2 + 100, car_y + 120), (100, 50, 50), -1)

            if abs(frame_idx - bump_frame) < 3:
                cv2.putText(img, "BUMP", (self.width // 2 - 80, self.height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 5)

            speed = 5 if frame_idx < bump_frame else 0
            cv2.putText(img, f"Speed: {speed} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            out.write(img)

        out.release()
        print(f"[OK] Parking bump video saved: {output_path}")
        return {
            "type": "low_parking_bump",
            "duration_sec": duration_sec,
            "severity": "LOW",
            "ground_truth": {"fault_ratio": 100.0, "scenario": "parking", "fraud_risk": 0.0},
        }

    def generate_swerve_avoidance_video(self, output_path: str, duration_sec: int = 30):
        """Generate swerve to avoid obstacle scenario."""
        print(f"Generating swerve avoidance video: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        total_frames = duration_sec * self.fps
        event_frame = int(15 * self.fps)

        for frame_idx in range(total_frames):
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.rectangle(img, (0, 0), (self.width, self.height // 2), (180, 140, 100), -1)
            road_y = self.height // 2
            cv2.rectangle(img, (0, road_y), (self.width, self.height), (80, 80, 80), -1)

            # Obstacle (debris/box on road)
            obs_y = road_y + 200
            cv2.rectangle(img, (self.width // 2 - 40, obs_y), (self.width // 2 + 40, obs_y + 50), (80, 60, 40), -1)

            # Camera swerve effect after event
            if abs(frame_idx - event_frame) < 30:
                offset = int(50 * np.sin((frame_idx - event_frame) * 0.3))
                M = np.float32([[1, 0, offset], [0, 1, 0]])
                img = cv2.warpAffine(img, M, (self.width, self.height))

            if abs(frame_idx - event_frame) < 20:
                cv2.putText(img, "! SWERVE !", (self.width // 2 - 200, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 165, 255), 6)

            speed = 60 if abs(frame_idx - event_frame) > 45 else max(20, int(60 * abs(frame_idx - event_frame) / 45))
            cv2.putText(img, f"Speed: {speed} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            out.write(img)

        out.release()
        print(f"[OK] Swerve avoidance video saved: {output_path}")
        return {
            "type": "swerve_avoidance",
            "duration_sec": duration_sec,
            "severity": "MEDIUM",
            "ground_truth": {"fault_ratio": 0.0, "scenario": "obstacle_avoidance", "fraud_risk": 0.0},
        }

    def generate_hard_braking_clear_video(self, output_path: str, duration_sec: int = 30):
        """Generate hard braking with no nearby vehicles scenario (LOW severity)."""
        print(f"Generating hard braking (clear) video: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        total_frames = duration_sec * self.fps
        brake_frame = int(15 * self.fps)

        for frame_idx in range(total_frames):
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.rectangle(img, (0, 0), (self.width, self.height // 2), (180, 140, 100), -1)
            road_y = self.height // 2
            cv2.rectangle(img, (0, road_y), (self.width, self.height), (80, 80, 80), -1)

            # Empty road ahead — no vehicles
            # Lane markings (animated)
            offset = (frame_idx * 5) % 100
            for lane_x in [self.width // 3, 2 * self.width // 3]:
                y = road_y
                while y < self.height:
                    y_start = y - offset
                    y_end = min(y_start + 50, self.height)
                    if y_start < self.height:
                        cv2.line(img, (lane_x, max(y_start, road_y)), (lane_x, y_end), (255, 255, 255), 8)
                    y += 100

            # Speed drops on braking
            if frame_idx < brake_frame:
                speed = 80
            elif frame_idx < brake_frame + 60:
                progress = (frame_idx - brake_frame) / 60
                speed = int(80 * (1 - progress))
            else:
                speed = 0

            cv2.putText(img, f"Speed: {speed} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            out.write(img)

        out.release()
        print(f"[OK] Hard braking (clear) video saved: {output_path}")
        return {
            "type": "hard_braking_clear",
            "duration_sec": duration_sec,
            "severity": "LOW",
            "ground_truth": {"fault_ratio": 0.0, "scenario": "hard_braking_clear", "fraud_risk": 0.0},
        }

    def _add_audio(self, video_path: str, audio_type: str):
        """Add synthetic audio track"""
        # For now, just create silent audio
        # In production, you would add:
        # - Engine sound
        # - Crash sound (for collision)
        # - Brake squeal (for near-miss)
        pass


def main():
    parser = argparse.ArgumentParser(description="Generate dashcam footage for insurance demo")
    parser.add_argument(
        "--output-dir", type=str, default="data/dashcam_demo", help="Output directory for generated videos"
    )
    parser.add_argument("--duration", type=int, default=30, help="Video duration in seconds")
    parser.add_argument("--resolution", type=str, default="1920x1080", help="Video resolution (WIDTHxHEIGHT)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")

    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split("x"))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = DashcamVideoGenerator(width=width, height=height, fps=args.fps)

    # Generate all videos
    print("\n" + "=" * 60)
    print("Generating Dashcam Demo Videos (10 scenarios)")
    print("=" * 60 + "\n")

    metadata = {}

    # HIGH severity
    collision_path = str(output_dir / "collision.mp4")
    metadata["collision"] = generator.generate_collision_video(collision_path, args.duration)

    collision_int_path = str(output_dir / "collision_intersection.mp4")
    metadata["collision_intersection"] = generator.generate_collision_intersection_video(
        collision_int_path, args.duration
    )

    # MEDIUM severity
    near_miss_path = str(output_dir / "near_miss.mp4")
    metadata["near_miss"] = generator.generate_near_miss_video(near_miss_path, args.duration)

    near_miss_cyclist_path = str(output_dir / "near_miss_cyclist.mp4")
    metadata["near_miss_cyclist"] = generator.generate_near_miss_cyclist_video(near_miss_cyclist_path, args.duration)

    near_miss_vehicle_path = str(output_dir / "near_miss_vehicle.mp4")
    metadata["near_miss_vehicle"] = generator.generate_near_miss_vehicle_video(near_miss_vehicle_path, args.duration)

    swerve_path = str(output_dir / "swerve_avoidance.mp4")
    metadata["swerve_avoidance"] = generator.generate_swerve_avoidance_video(swerve_path, args.duration)

    # NONE severity
    normal_path = str(output_dir / "normal.mp4")
    metadata["normal"] = generator.generate_normal_video(normal_path, args.duration)

    normal_city_path = str(output_dir / "normal_city.mp4")
    metadata["normal_city"] = generator.generate_normal_city_video(normal_city_path, args.duration)

    # LOW severity
    parking_path = str(output_dir / "low_parking_bump.mp4")
    metadata["low_parking_bump"] = generator.generate_low_parking_bump_video(parking_path, args.duration)

    hard_braking_path = str(output_dir / "hard_braking_clear.mp4")
    metadata["hard_braking_clear"] = generator.generate_hard_braking_clear_video(hard_braking_path, args.duration)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("[SUCCESS] All 10 videos generated successfully!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("  HIGH:   collision.mp4, collision_intersection.mp4")
    print("  MEDIUM: near_miss.mp4, near_miss_cyclist.mp4, near_miss_vehicle.mp4, swerve_avoidance.mp4")
    print("  NONE:   normal.mp4, normal_city.mp4")
    print("  LOW:    low_parking_bump.mp4, hard_braking_clear.mp4")
    print("  metadata.json  (ground truth)")
    print()


if __name__ == "__main__":
    main()
