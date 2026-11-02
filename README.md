# Multimodal Attendance & Engagement Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8.svg)](https://opencv.org/)
[![Privacy Compliant](https://img.shields.io/badge/Privacy-FERPA%20%2F%20GDPR%20Ready-green.svg)](#privacy--ethical-ai-framework)
[![Tests Passing](https://img.shields.io/badge/Tests-11%2F11%20Passing-brightgreen.svg)](tests/)

An AI-powered, privacy-conscious classroom attendance and participation telemetry system. Combines computer vision (sustained face tracking and consented identification) with acoustic processing (speech-silence dynamics and classroom interaction index) to assist educators with automated attendance and pedagogically actionable analytics.

---

## Key Features

1. **Telemetry Dashboard**
   - Live metrics: Total students, classes conducted, historical attendance rate %, and engagement scores.
   - Interactive Plotly visualizations for session attendance comparisons and speech/silence distribution.
   - Recent activity feed with one-click drilldown into past classroom sessions.

2. **Student Management & Biometric Consent**
   - Full student directory with unique IDs, grades, sections, and guardian authorizations.
   - **Consent-First Architecture:** Facial biometric feature embeddings are only extracted and stored for students who have affirmative consent on file.
   - **One-Click Biometric Purge:** Immediate and permanent deletion of biometric representations upon consent revocation.

3. **Automated Attendance Tracking**
   - Upload classroom video recordings (MP4, AVI, MOV) or capture live camera snapshots.
   - **Centroid & IoU Face Tracking:** Distinguishes students across frames, tracking continuous observation duration and movement energy.
   - **Threshold Gating:** Configurable observation duration (e.g. &ge; 5.0 seconds) and face match confidence (&ge; 65%) to eliminate transient false-positives.
   - **Anti-Duplicate Guarantee:** SQLite schema enforces `UNIQUE(session_id, student_id)` to prevent duplicate records during the same session.
   - **Teacher Override Interface:** Instructors can review, modify (Present, Late, Absent, Excused), and annotate records with audit notes.

4. **Classroom Acoustic Telemetry**
   - Analyzes speech activity, quiet intervals, and energy dynamics using SoundFile and Librosa.
   - Voice Activity Detection (VAD) with adaptive noise floor estimation.
   - Computes Speech Ratio %, Silence Ratio %, Discussion Intensity Index, and Speaker Turn Events.
   - **Strict Privacy Guardrail:** Operates exclusively at the *aggregate classroom level*. Does NOT identify individual voices or use audio as standalone proof of attendance.

5. **Multimodal Fusion & Explainable AI**
   - Fuses visual observation duration, facial match confidence, and acoustic classroom context.
   - Every attendance entry generates an **Explainability Card** detailing why a student was marked Present or Absent.
   - Never treats general classroom noise as proof that a specific student is present.

6. **Cautious Engagement Analytics**
   - Second-by-second temporal engagement timeline aligning visible faces, movement stability, and speech bursts.
   - Automatically categorizes pedagogical states: *Interactive Discussion*, *Attentive Lecture / Focus*, *Silent Activity*, and *Transition*.
   - **Pedagogical Advisory Insights:** Detects sustained quiet windows (&gt;25s) or high lecture stretches to suggest formative check-ins.
   - **Ethical AI Disclaimer:** Explicitly notes that engagement metrics are behavioral proxies and do NOT measure students' inner thoughts, feelings, or comprehension.

7. **Reports & Data Export**
   - Student-level historical attendance summaries and session registers.
   - One-click CSV exports for school information systems.
   - **De-identified Research Export:** Exports anonymized datasets using SHA-256 cryptographic pseudo-tokens (`Participant_abc123`) for ethical educational research.

8. **Privacy, Retention & Audit Trail**
   - Configurable retention periods (30, 60, 90, 180, 365 days) with automated database purging.
   - Immutable chronological audit trail recording all biometric registrations, deletions, overrides, and purges.

---

## Project Structure

```text
multimodal-attendance-agent/
├── app.py                      # Master Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation & setup guide
├── .gitignore                  # Git exclusion rules
├── config.py                   # Centralized application configurations
├── verify_env.py               # Dependency and environment verification script
├── run_tests.py                # Standalone test runner
├── database/
│   ├── __init__.py
│   ├── schema.sql              # SQLite relational schema with constraints
│   ├── db_init.py              # CLI database initialization utility
│   └── db_manager.py           # Thread-safe database manager & CRUD operations
├── modules/
│   ├── __init__.py
│   ├── audio_module.py         # Acoustic feature extraction & Voice Activity Detection
│   ├── multimodal_fusion.py    # Evidence fusion, explainability, & advisory logic
│   ├── student_service.py      # Student profile & biometric consent management
│   ├── attendance_service.py   # Automated attendance tracking & manual override
│   ├── reporting_service.py    # CSV export & tabular aggregations
│   └── privacy_service.py      # Retention purging, de-identification, & audit trail
├── vision/
│   ├── __init__.py
│   ├── face_detector.py        # OpenCV face detector with CLAHE normalization
│   ├── tracker.py              # Centroid + IoU face tracking & velocity estimation
│   ├── recognizer.py           # Feature embedding encoder (OpenCV fallback + DeepFace)
│   ├── attendance.py           # Temporal threshold qualification evaluator
│   └── pipeline.py             # Orchestrates frame/video processing
├── sample_data/
│   ├── fictional_students.json # Seed roster of fictional students
│   ├── seed_database.py        # Populates SQLite with fictional students & sessions
│   └── generate_sample_media.py# Generates sample classroom video, audio, & avatars
├── tests/
│   ├── __init__.py
│   ├── test_database.py        # SQLite schema & duplicate constraint tests
│   ├── test_vision_module.py   # Vision detector, tracker, & recognizer tests
│   ├── test_audio_module.py    # Audio VAD & RMS energy tests
│   └── test_multimodal_fusion.py # Fusion explainability & timeline tests
└── data/                       # Local SQLite DB, sample media, and avatars
```

---

## Getting Started (Windows & VS Code)

### Prerequisites

- **Python 3.10+** (Python 3.11, 3.12, 3.14 tested and supported)
- **Git** (optional, for version control)
- **Visual Studio Code** (recommended IDE)

### 1. Open the Project in VS Code

Open VS Code and navigate to the project directory:
```powershell
cd C:\Users\Tanisha\.gemini\antigravity-ide\scratch\multimodal-attendance-agent
code .
```

### 2. Create and Activate Virtual Environment

In the VS Code terminal (PowerShell):
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

> **Note for PowerShell execution policy:** If script execution is restricted, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 3. Install Dependencies

Install the required packages from `requirements.txt`:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify your environment setup:
```powershell
python verify_env.py
```

### 4. Initialize Database with Sample Data

Run the database setup script with fictional sample students and historical sessions:
```powershell
python database/db_init.py --with-sample-data
```

To regenerate the synthetic classroom video and audio test files:
```powershell
python sample_data/generate_sample_media.py
```

### 5. Run the Application

Launch the Streamlit dashboard:
```powershell
streamlit run app.py
```

The application will automatically open in your default browser at:
`http://localhost:8501`

---

## Running the Automated Test Suite

Run the full unit and integration test suite (11 unit tests covering database constraints, vision tracking, audio VAD, and multimodal fusion):

```powershell
# Run using the standalone runner
python run_tests.py

# Or directly with pytest
pytest tests/ -v
```

All 11 tests validate:
- Anti-duplicate database constraints and manual override logging.
- Biometric purge upon student consent revocation.
- Face detection, centroid displacement tracking, and deregistration.
- Normalized L2 feature vector extraction and cosine similarity matching.
- Minimum observation duration thresholding.
- Audio sliding-frame RMS energy and Voice Activity Detection (VAD).
- Multimodal explainability card generation and pedagogical advisory rules.

---

## Privacy & Ethical AI Framework

This project was engineered following ethical AI principles for educational environments:

| Principle | Technical Implementation |
| :--- | :--- |
| **Affirmative Consent** | Biometric facial vectors are only extracted and stored when `consent_enrolled = 1`. Unconsented students are never processed for facial identification. |
| **Immediate Biometric Purge** | When consent is revoked, `purge_biometrics_for_student()` immediately zeroes out and permanently deletes all stored embeddings. |
| **Auditing & Traceability** | Every biometric enrollment, status modification, manual teacher override, and retention purge is logged in `privacy_audit_logs`. |
| **Aggregate-Only Audio** | Acoustic analysis runs strictly across the room as an aggregate signal (RMS, speech ratio, turn transitions). Audio is never used to identify individual speakers. |
| **Explainable Decisions** | Attendance records include a transparent mathematical explanation (observation duration, frame count, match confidence %, and context). |
| **Non-Diagnostic Analytics** | Engagement telemetry is strictly labeled as behavioral proxies (motion, vocal bursts) and makes no claims regarding internal comprehension or emotions. |
| **Data Retention Controls** | Configurable retention window (e.g. 90 days) with automated purge routines to avoid indefinite storage of student records. |

---

## Model Fallbacks & Offline Operation

- **Face Recognition:** Includes a built-in, lightweight, 128-dimensional OpenCV HOG/spatial descriptor that executes locally in milliseconds without downloading heavy weights. Optionally supports DeepFace (`Facenet512` / `VGG-Face`) when configured.
- **Audio Analysis:** Powered by `soundfile` and `librosa` with robust fallback algorithms for Fourier transform and energy calculation.
- **Zero Cloud Leakage:** All computation and storage (SQLite) happens locally on the teacher's workstation. No external API keys or cloud connections are required.
