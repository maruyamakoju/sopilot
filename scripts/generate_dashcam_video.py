"""
Generate realistic dashcam footage for insurance demo

Creates 3 types of videos:
1. Collision (rear-end accident)
2. Near-miss (sudden braking)
3. Normal driving

High-quality, professional-grade for demo purposes
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import subprocess
import json


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

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                "fraud_risk": 0.0
            }
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

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                "fraud_risk": 0.0
            }
        }

    def generate_normal_video(self, output_path: str, duration_sec: int = 30):
        """
        Generate normal driving scenario

        Timeline:
        0-30s: Normal highway driving, no incidents
        """
        print(f"Generating normal driving video: {output_path}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
            "ground_truth": {
                "fault_ratio": 0.0,
                "scenario": "normal_driving",
                "fraud_risk": 0.0
            }
        }

    def _create_highway_frame(self, frame_idx: int, total_frames: int, collision_frame: int):
        """Create realistic highway frame with approaching vehicle"""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Sky (gradient)
        sky_color_top = (180, 120, 60)  # Light blue
        sky_color_bottom = (220, 180, 140)
        for y in range(self.height // 2):
            ratio = y / (self.height // 2)
            color = tuple(int(sky_color_top[i] * (1-ratio) + sky_color_bottom[i] * ratio) for i in range(3))
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
        cv2.rectangle(img, (car_x + 30, window_y), (car_x + car_width - 30, window_y + window_height), (100, 150, 200), -1)

        # Brake lights (RED) if approaching collision
        if frame_idx > collision_frame - 150:  # 5 seconds before collision
            brake_intensity = min(255, int(255 * (collision_frame - frame_idx) / -150))
            brake_color = (0, 0, min(255, 150 + brake_intensity))
            tail_light_y = car_y + car_height - 30
            cv2.rectangle(img, (car_x + 10, tail_light_y), (car_x + 40, tail_light_y + 20), brake_color, -1)
            cv2.rectangle(img, (car_x + car_width - 40, tail_light_y), (car_x + car_width - 10, tail_light_y + 20), brake_color, -1)

        # Collision effect (white flash + impact)
        if abs(frame_idx - collision_frame) < 5:
            flash_intensity = 255 - abs(frame_idx - collision_frame) * 50
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (255, 255, 255), -1)
            cv2.addWeighted(overlay, flash_intensity / 255.0, img, 1 - flash_intensity / 255.0, 0, img)

            # Impact text
            cv2.putText(img, "COLLISION!", (self.width // 2 - 200, self.height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)

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
            cv2.putText(img, "! PEDESTRIAN !", (self.width // 2 - 250, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)

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

    def _add_audio(self, video_path: str, audio_type: str):
        """Add synthetic audio track"""
        # For now, just create silent audio
        # In production, you would add:
        # - Engine sound
        # - Crash sound (for collision)
        # - Brake squeal (for near-miss)
        pass


def main():
    parser = argparse.ArgumentParser(description='Generate dashcam footage for insurance demo')
    parser.add_argument('--output-dir', type=str, default='data/dashcam_demo',
                       help='Output directory for generated videos')
    parser.add_argument('--duration', type=int, default=30,
                       help='Video duration in seconds')
    parser.add_argument('--resolution', type=str, default='1920x1080',
                       help='Video resolution (WIDTHxHEIGHT)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second')

    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split('x'))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = DashcamVideoGenerator(width=width, height=height, fps=args.fps)

    # Generate 3 videos
    print("\n" + "="*60)
    print("Generating Dashcam Demo Videos")
    print("="*60 + "\n")

    metadata = {}

    # 1. Collision
    collision_path = str(output_dir / "collision.mp4")
    metadata['collision'] = generator.generate_collision_video(collision_path, args.duration)

    # 2. Near-miss
    near_miss_path = str(output_dir / "near_miss.mp4")
    metadata['near_miss'] = generator.generate_near_miss_video(near_miss_path, args.duration)

    # 3. Normal
    normal_path = str(output_dir / "normal.mp4")
    metadata['normal'] = generator.generate_normal_video(normal_path, args.duration)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print("[SUCCESS] All videos generated successfully!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - collision.mp4  (HIGH severity)")
    print(f"  - near_miss.mp4  (MEDIUM severity)")
    print(f"  - normal.mp4     (NONE severity)")
    print(f"  - metadata.json  (ground truth)")
    print()


if __name__ == '__main__':
    main()
