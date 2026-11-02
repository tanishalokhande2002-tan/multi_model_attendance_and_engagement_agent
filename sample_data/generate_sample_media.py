"""
Sample Media Generator for Multimodal Attendance & Engagement Agent.
Generates synthetic classroom video (MP4) and audio (WAV) files
for immediate offline testing without private student biometric data.
"""

import math
from pathlib import Path
import numpy as np
import cv2
import soundfile as sf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AVATAR_DIR = DATA_DIR / "avatars"


def generate_sample_audio(filename: str = "sample_classroom.wav", duration_sec: float = 10.0, sr: int = 16000) -> Path:
    """
    Synthesize a multi-phase classroom audio track:
    - 0-3s: Teacher lecture vocal cadence (modulated harmonic tones)
    - 3-5s: Silence / reflection interval
    - 5-8s: Student question and discussion burst
    - 8-10s: Teacher response
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / filename
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    audio = np.zeros(total_samples, dtype=np.float32)

    # 1. Background classroom ambient room tone (pink-like noise)
    noise = np.random.normal(0, 0.003, total_samples).astype(np.float32)

    # 2. Segment 1: Teacher lecture (0 to 3s)
    mask1 = (t >= 0.5) & (t < 3.2)
    t1 = t[mask1]
    # Fundamental pitch ~ 160Hz with natural speech harmonics (320Hz, 480Hz, 1200Hz)
    pitch_mod = 160.0 + 15.0 * np.sin(2 * np.pi * 3.0 * t1)
    phase1 = np.cumsum(2 * np.pi * pitch_mod / sr)
    speech1 = (
        0.04 * np.sin(phase1) +
        0.025 * np.sin(2 * phase1) +
        0.015 * np.sin(3 * phase1) +
        0.008 * np.sin(8 * phase1)
    )
    # Speech syllable envelope modulation (4 syllables per second)
    env1 = np.maximum(0.0, np.sin(2 * np.pi * 4.0 * t1)) ** 2
    audio[mask1] += (speech1 * env1).astype(np.float32)

    # 3. Segment 2: Pause / reflection (3.2 to 5.0s) -> Only ambient room noise

    # 4. Segment 3: Student question (5.0 to 7.8s)
    mask2 = (t >= 5.0) & (t < 7.8)
    t2 = t[mask2]
    # Higher fundamental pitch ~ 220Hz
    pitch_mod2 = 220.0 + 20.0 * np.sin(2 * np.pi * 4.5 * t2)
    phase2 = np.cumsum(2 * np.pi * pitch_mod2 / sr)
    speech2 = (
        0.035 * np.sin(phase2) +
        0.02 * np.sin(2 * phase2) +
        0.01 * np.sin(4 * phase2)
    )
    env2 = np.maximum(0.0, np.sin(2 * np.pi * 5.0 * t2)) ** 2
    audio[mask2] += (speech2 * env2).astype(np.float32)

    # 5. Segment 4: Brief teacher wrap-up (8.2 to 9.8s)
    mask3 = (t >= 8.2) & (t < 9.8)
    t3 = t[mask3]
    phase3 = np.cumsum(2 * np.pi * 160.0 / sr)
    speech3 = 0.03 * np.sin(phase3)
    env3 = np.maximum(0.0, np.sin(2 * np.pi * 3.5 * t3)) ** 2
    audio[mask3] += (speech3 * env3).astype(np.float32)

    # Combine with room ambience and normalize
    full_audio = audio + noise
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val * 0.75

    sf.write(str(out_path), full_audio, sr)
    print(f"[OK] Generated sample audio at: {out_path}")
    return out_path


def generate_sample_video(filename: str = "sample_classroom.mp4", duration_sec: float = 6.0, fps: int = 15) -> Path:
    """
    Synthesize an illustrative classroom video:
    Renders classroom desks with animated student avatars, natural breathing/head motion,
    and a teacher whiteboard to allow the vision pipeline to detect and track faces.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / filename
    width, height = 640, 360
    total_frames = int(duration_sec * fps)

    # Define video codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # Student anchor positions across 2 classroom rows
    student_nodes = [
        {"id": "STU-1001", "name": "Alex Chen", "base_x": 120, "base_y": 140, "color": (190, 160, 140)},
        {"id": "STU-1002", "name": "Maya Patel", "base_x": 260, "base_y": 140, "color": (160, 140, 190)},
        {"id": "STU-1003", "name": "Liam Johnson", "base_x": 400, "base_y": 140, "color": (150, 180, 160)},
        {"id": "STU-1004", "name": "Sofia Rodriguez", "base_x": 520, "base_y": 140, "color": (180, 150, 170)},
        {"id": "STU-1005", "name": "Marcus Washington", "base_x": 180, "base_y": 240, "color": (160, 170, 190)},
        {"id": "STU-1006", "name": "Priya Sharma", "base_x": 340, "base_y": 240, "color": (190, 170, 150)},
        {"id": "STU-1007", "name": "Ethan Brooks", "base_x": 480, "base_y": 240, "color": (170, 190, 160)},
    ]

    for frame_idx in range(total_frames):
        t_sec = frame_idx / fps
        # Classroom background (subtle gradient wall and floor)
        frame = np.full((height, width, 3), (242, 240, 236), dtype=np.uint8)

        # Draw classroom whiteboard / presentation screen
        cv2.rectangle(frame, (160, 20), (480, 80), (255, 255, 255), -1)
        cv2.rectangle(frame, (160, 20), (480, 80), (70, 70, 70), 2)
        cv2.putText(frame, "Data Structures: Binary Search Trees", (175, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)

        # Draw desks
        cv2.rectangle(frame, (80, 180), (580, 195), (140, 110, 85), -1)
        cv2.rectangle(frame, (120, 280), (560, 295), (140, 110, 85), -1)

        # Draw each student with realistic subtle head motion
        for stu in student_nodes:
            # Natural micro-motion (breathing + nodding)
            dx = int(4.0 * math.sin(t_sec * 2.0 + stu["base_x"]))
            dy = int(3.0 * math.cos(t_sec * 1.5 + stu["base_y"]))
            cx = stu["base_x"] + dx
            cy = stu["base_y"] + dy

            # Student body / shoulders
            cv2.ellipse(frame, (cx, cy + 45), (32, 22), 0, 0, 180, stu["color"], -1)

            # Head / Face oval
            face_w, face_h = 22, 26
            cv2.ellipse(frame, (cx, cy), (face_w, face_h), 0, 0, 360, (220, 195, 175), -1)
            cv2.ellipse(frame, (cx, cy), (face_w, face_h), 0, 0, 360, (160, 130, 110), 1)

            # Hair
            cv2.ellipse(frame, (cx, cy - 12), (face_w + 1, 14), 0, 180, 360, (60, 45, 35), -1)

            # Eyes
            cv2.circle(frame, (cx - 7, cy - 3), 2, (50, 40, 30), -1)
            cv2.circle(frame, (cx + 7, cy - 3), 2, (50, 40, 30), -1)

            # Mouth
            cv2.ellipse(frame, (cx, cy + 10), (5, 2), 0, 0, 180, (120, 80, 70), 1)

        out.write(frame)

    out.release()
    print(f"[OK] Generated sample classroom video at: {out_path}")
    return out_path


def generate_sample_avatars() -> None:
    """Generate sample fictional avatar images for UI profile display."""
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    palette = [
        ("STU-1001", "AC", (180, 120, 70)),
        ("STU-1002", "MP", (70, 130, 180)),
        ("STU-1003", "LJ", (120, 160, 80)),
        ("STU-1004", "SR", (170, 70, 140)),
        ("STU-1005", "MW", (90, 110, 180)),
        ("STU-1006", "PS", (190, 140, 60)),
        ("STU-1007", "EB", (60, 170, 130)),
        ("STU-1008", "CK", (150, 80, 170)),
        ("STU-1009", "NA", (140, 140, 140)),
        ("STU-1010", "ED", (200, 100, 90)),
    ]

    for sid, initials, bg_color in palette:
        img = np.full((100, 100, 3), bg_color, dtype=np.uint8)
        # Rounded look
        cv2.circle(img, (50, 50), 46, (255, 255, 255), 2)
        # Center initials
        (tw, th), _ = cv2.getTextSize(initials, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(
            img,
            initials,
            ((100 - tw) // 2, (100 + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        avatar_path = AVATAR_DIR / f"{sid}.jpg"
        cv2.imwrite(str(avatar_path), img)

    print(f"[OK] Generated {len(palette)} fictional avatar icons at: {AVATAR_DIR}")


def generate_all() -> None:
    print("=" * 60)
    print("Generating Synthetic Classroom Media Files...")
    print("=" * 60)
    generate_sample_audio()
    generate_sample_video()
    generate_sample_avatars()
    print("=" * 60)
    print("Synthetic sample media generation complete.")


if __name__ == "__main__":
    generate_all()
