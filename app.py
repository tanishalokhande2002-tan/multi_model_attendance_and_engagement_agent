"""
Multimodal Attendance & Engagement Agent - Streamlit Application
A modern, privacy-preserving AI assistant for classroom attendance and aggregate engagement monitoring.
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import DatabaseManager
from vision.pipeline import VisionPipeline
from vision.recognizer import FaceRecognizer
from modules.audio_module import AudioProcessor
from modules.multimodal_fusion import MultimodalFusionEngine
from modules.student_service import StudentService
from modules.attendance_service import AttendanceService
from modules.reporting_service import ReportingService
from modules.privacy_service import PrivacyService
from sample_data.seed_database import seed_all

# =============================================================================
# Page Configuration & Styling
# =============================================================================

st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-title {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
    }
    .metric-card h4 {
        margin: 0;
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.3rem;
    }
    .metric-card .delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    .ethical-banner {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.9rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.2rem;
        font-size: 0.88rem;
        color: #1E40AF;
    }
    
    .privacy-banner {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 0.9rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1.2rem;
        font-size: 0.88rem;
        color: #166534;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# State & Service Initialization
# =============================================================================

@st.cache_resource
def get_services():
    """Initialize singleton instances of database and business services."""
    db = DatabaseManager()
    recognizer = FaceRecognizer(similarity_threshold=0.65)
    vision_pipeline = VisionPipeline(recognizer=recognizer)
    audio_processor = AudioProcessor()
    fusion_engine = MultimodalFusionEngine()
    
    student_svc = StudentService(db=db, recognizer=recognizer)
    attendance_svc = AttendanceService(
        db=db,
        vision_pipeline=vision_pipeline,
        audio_processor=audio_processor,
        fusion_engine=fusion_engine,
    )
    reporting_svc = ReportingService(db=db)
    privacy_svc = PrivacyService(db=db)

    # Automatically check if sample data is needed on very first boot
    stats = db.get_dashboard_stats()
    if stats["total_students"] == 0:
        seed_all(db)

    return {
        "db": db,
        "student_svc": student_svc,
        "attendance_svc": attendance_svc,
        "reporting_svc": reporting_svc,
        "privacy_svc": privacy_svc,
        "vision_pipeline": vision_pipeline,
        "audio_processor": audio_processor,
        "fusion_engine": fusion_engine,
        "recognizer": recognizer,
    }

services = get_services()
db = services["db"]
student_svc = services["student_svc"]
attendance_svc = services["attendance_svc"]
reporting_svc = services["reporting_svc"]
privacy_svc = services["privacy_svc"]


# =============================================================================
# Sidebar Navigation
# =============================================================================

st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1rem 0;">
    <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #1E293B;">Multimodal Agent</h2>
    <span style="font-size: 0.75rem; background: #E0E7FF; color: #3730A3; padding: 2px 8px; border-radius: 12px; font-weight: 600;">v2.0 AI Classroom</span>
</div>
""", unsafe_allow_html=True)

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dashboard Overview",
        "👥 Student Management",
        "📷 Take Attendance",
        "🎙️ Audio Analysis",
        "🔄 Multimodal Fusion",
        "📈 Engagement Analytics",
        "📑 Reports & Export",
        "⚙️ Privacy & Settings",
    ],
    index=0,
)

st.sidebar.markdown("---")
# Sidebar Quick Stats & Controls
stats = db.get_dashboard_stats()
st.sidebar.caption("System Status")
st.sidebar.markdown(f"**Students Enrolled:** {stats['total_students']} ({stats['consented_students']} Consented)")
st.sidebar.markdown(f"**Sessions Recorded:** {stats['total_sessions']}")
st.sidebar.markdown(f"**Average Attendance:** {stats['overall_attendance_pct']}%")

if st.sidebar.button("🔄 Reset / Re-seed Sample Data"):
    seed_all(db)
    st.sidebar.success("Sample database re-seeded successfully!")
    st.rerun()


# =============================================================================
# View 1: Dashboard Overview
# =============================================================================

if nav_choice == "📊 Dashboard Overview":
    st.markdown('<div class="main-title">Classroom Attendance & Engagement Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multimodal AI telemetry synthesizing visual attendance and acoustic participation</div>', unsafe_allow_html=True)

    # 4 Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Registered</h4>
            <div class="value">{stats['total_students']}</div>
            <div class="delta" style="color: #16A34A;">{stats['consented_students']} with Biometric Consent</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Sessions Logged</h4>
            <div class="value">{stats['total_sessions']}</div>
            <div class="delta" style="color: #2563EB;">Across 2 Grade Levels</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Attendance Rate</h4>
            <div class="value">{stats['overall_attendance_pct']}%</div>
            <div class="delta" style="color: #16A34A;">Target: &ge; 80%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Engagement Index</h4>
            <div class="value">{stats['avg_engagement_pct']}%</div>
            <div class="delta" style="color: #D97706;">Moderate to High</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Charts
    col_chart1, col_chart2 = st.columns([1.2, 1])
    
    with col_chart1:
        st.subheader("Historical Attendance by Session")
        sessions = db.get_all_sessions(limit=8)
        if sessions:
            sess_names = [f"{s['course_name'][:20]}.. ({s['session_date']})" for s in reversed(sessions)]
            sess_ids = [s["session_id"] for s in reversed(sessions)]
            
            present_rates = []
            for sid in sess_ids:
                att_list = db.get_session_attendance(sid)
                if att_list:
                    p = sum(1 for a in att_list if a["status"] in ("Present", "Late"))
                    present_rates.append(round(p / len(att_list) * 100.0, 1))
                else:
                    present_rates.append(85.0)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sess_names,
                y=present_rates,
                marker_color="#3B82F6",
                text=[f"{r}%" for r in present_rates],
                textposition="auto",
            ))
            fig.update_layout(
                yaxis=dict(title="Attendance Rate %", range=[0, 105]),
                xaxis=dict(tickangle=-25),
                height=320,
                margin=dict(l=20, r=20, t=20, b=60),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recorded sessions yet.")

    with col_chart2:
        st.subheader("Acoustic Participation Distribution")
        # Fetch audio metrics for recent sessions
        sessions = db.get_all_sessions(limit=6)
        speech_ratios = []
        silence_ratios = []
        for s in sessions:
            m = db.get_audio_metrics(s["session_id"])
            if m:
                speech_ratios.append(m["speech_ratio"])
                silence_ratios.append(m["silence_ratio"])
            else:
                speech_ratios.append(0.5)
                silence_ratios.append(0.5)

        avg_speech = np.mean(speech_ratios) if speech_ratios else 0.5
        avg_silence = np.mean(silence_ratios) if silence_ratios else 0.5

        fig_pie = go.Figure(data=[go.Pie(
            labels=["Active Classroom Discussion / Q&A", "Silent Study / Contemplation"],
            values=[round(avg_speech * 100, 1), round(avg_silence * 100, 1)],
            hole=0.55,
            marker_colors=["#10B981", "#94A3B8"],
        )])
        fig_pie.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Recent Sessions Table
    st.subheader("Recent Classroom Sessions")
    if sessions:
        rec_data = []
        for s in sessions:
            rec_data.append({
                "Session ID": s["session_id"],
                "Course Name": s["course_name"],
                "Instructor": s.get("instructor_name", "-"),
                "Grade / Sec": f"{s.get('grade', '')} {s.get('section', '')}".strip(),
                "Date": s["session_date"],
                "Status": s.get("status", "Completed"),
            })
        st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)


# =============================================================================
# View 2: Student Management
# =============================================================================

elif nav_choice == "👥 Student Management":
    st.markdown('<div class="main-title">Student Management & Biometric Enrollment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Maintain student records with strict consent requirements and privacy safeguards</div>', unsafe_allow_html=True)

    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📋 View Student Roster", "➕ Register New Student", "🛡️ Biometric Consent & Revocation"])

    with sub_tab1:
        st.subheader("Enrolled Students Directory")
        c1, c2 = st.columns([1, 1])
        with c1:
            filter_grade = st.selectbox("Filter by Grade", ["All Grades", "Grade 10", "Grade 11"], index=0)
        with c2:
            filter_sec = st.selectbox("Filter by Section", ["All Sections", "A", "B"], index=0)

        f_grade = None if filter_grade == "All Grades" else filter_grade
        f_sec = None if filter_sec == "All Sections" else filter_sec

        students = student_svc.get_all_students(grade=f_grade, section=f_sec)
        if students:
            # Render grid of student cards
            cols = st.columns(3)
            for i, stu in enumerate(students):
                col = cols[i % 3]
                with col:
                    consent_badge = "🟢 Consented" if stu.get("consent_enrolled") else "🔴 Withheld"
                    face_badge = "📸 Face Enrolled" if stu.get("face_encoding") else "⚪ No Face Data"
                    
                    st.markdown(f"""
                    <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; background: #FFFFFF;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="font-size: 1.05rem; color: #1E293B;">{stu['name']}</strong>
                            <span style="font-size: 0.75rem; background: #F1F5F9; padding: 2px 6px; border-radius: 4px;">{stu['student_id']}</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #64748B; margin: 4px 0;">{stu.get('grade', '')} - Sec {stu.get('section', '')} | {stu.get('email', '')}</div>
                        <div style="margin-top: 8px; font-size: 0.78rem;">
                            <span>{consent_badge}</span> &bull; <span>{face_badge}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"Manage {stu['name']}"):
                        if st.button("Delete Student", key=f"del_{stu['student_id']}"):
                            ok, msg = student_svc.delete_student(stu["student_id"])
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
        else:
            st.info("No students found matching current filters.")

    with sub_tab2:
        st.subheader("Add Student with Consent Verification")
        st.markdown("""
        <div class="privacy-banner">
            <strong>Ethical AI Guardrail:</strong> Biometric facial features are ONLY stored after affirmative consent. 
            Students who withhold consent will still be in the class roster for manual or alternate attendance.
        </div>
        """, unsafe_allow_html=True)

        with st.form("add_student_form"):
            c_id, c_name = st.columns(2)
            with c_id:
                new_id = st.text_input("Unique Student ID (e.g. STU-1011)", value=f"STU-{1000 + stats['total_students'] + 1}")
            with c_name:
                new_name = st.text_input("Full Name", placeholder="e.g. Jordan Miller")

            c_email, c_grade, c_sec = st.columns(3)
            with c_email:
                new_email = st.text_input("Email", placeholder="student@school.edu")
            with c_grade:
                new_grade = st.selectbox("Grade Level", ["Grade 9", "Grade 10", "Grade 11", "Grade 12"], index=1)
            with c_sec:
                new_sec = st.selectbox("Section", ["A", "B", "C"], index=0)

            st.markdown("##### Consent & Authorization")
            consent_cb = st.checkbox("Student / Guardian has signed the biometric privacy consent form", value=True)
            guardian_cb = st.checkbox("Guardian authorization verified (for minors)", value=True)

            photo_file = st.file_uploader("Upload Reference Face Photo (Optional)", type=["jpg", "jpeg", "png"])

            submitted = st.form_submit_button("Register Student")
            if submitted:
                photo_bgr = None
                if photo_file:
                    file_bytes = np.asarray(bytearray(photo_file.read()), dtype=np.uint8)
                    photo_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                ok, msg = student_svc.register_student(
                    student_id=new_id,
                    name=new_name,
                    email=new_email,
                    grade=new_grade,
                    section=new_sec,
                    consent_enrolled=consent_cb,
                    guardian_consent=guardian_cb,
                    photo_bgr=photo_bgr,
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with sub_tab3:
        st.subheader("Biometric Consent & Purge Operations")
        st.markdown("""
        <div class="privacy-banner">
            When a student revokes biometric consent, all facial representations are <strong>immediately and permanently deleted</strong> from the database.
        </div>
        """, unsafe_allow_html=True)

        all_stu = student_svc.get_all_students()
        if all_stu:
            stu_options = {f"{s['name']} ({s['student_id']})": s["student_id"] for s in all_stu}
            selected_stu_label = st.selectbox("Select Student to Update Consent", list(stu_options.keys()))
            selected_sid = stu_options[selected_stu_label]
            target_student = db.get_student(selected_sid)

            current_consent = bool(target_student.get("consent_enrolled", 0))
            st.write(f"Current Status: **{'Authorized' if current_consent else 'Consent Withheld'}**")
            has_bio = bool(target_student.get("face_encoding"))
            st.write(f"Biometric Vector Stored: **{'Yes (128-d)' if has_bio else 'None'}**")

            col_rev, col_auth = st.columns(2)
            with col_rev:
                if st.button("🚫 Revoke Consent & Purge Biometrics", type="secondary"):
                    ok, msg = student_svc.update_student_consent(selected_sid, consent_enrolled=False)
                    st.warning(msg)
                    st.rerun()

            with col_auth:
                if st.button("✅ Authorize Biometric Consent", type="primary"):
                    ok, msg = student_svc.update_student_consent(selected_sid, consent_enrolled=True, guardian_consent=True)
                    st.success(msg)
                    st.rerun()


# =============================================================================
# View 3: Take Attendance
# =============================================================================

elif nav_choice == "📷 Take Attendance":
    st.markdown('<div class="main-title">Automated Attendance Tracking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload classroom video or use camera input to track sustained presence and match enrolled faces</div>', unsafe_allow_html=True)

    c_sess1, c_sess2, c_sess3 = st.columns(3)
    with c_sess1:
        course_name = st.text_input("Course Name", value="Physics 101 - Lecture 4")
    with c_sess2:
        instructor_name = st.text_input("Instructor", value="Dr. Marie Curie")
    with c_sess3:
        sess_grade = st.selectbox("Grade & Section", ["Grade 10 - Sec A", "Grade 10 - Sec B", "Grade 11 - Sec A"], index=0)

    # Configurable parameters expander
    with st.expander("⚙️ Advanced Detection & Attendance Thresholds", expanded=False):
        c_th1, c_th2 = st.columns(2)
        with c_th1:
            min_duration = st.slider(
                "Minimum Observation Duration (Seconds)",
                min_value=2.0, max_value=20.0, value=5.0, step=0.5,
                help="Sustained continuous or aggregate visibility required to confirm attendance candidate.",
            )
        with c_th2:
            conf_thresh = st.slider(
                "Face Match Confidence Threshold",
                min_value=0.50, max_value=0.90, value=0.65, step=0.05,
                help="Minimum cosine similarity required between detected face and enrolled reference.",
            )

    modality_tab1, modality_tab2 = st.tabs(["📁 Upload Classroom Video", "📷 Live Camera Input"])

    with modality_tab1:
        st.subheader("Process Video Recording")
        col_up, col_sample = st.columns([2, 1])
        
        sample_video_path = PROJECT_ROOT / "data" / "sample_classroom.mp4"
        sample_audio_path = PROJECT_ROOT / "data" / "sample_classroom.wav"

        with col_up:
            uploaded_video = st.file_uploader("Upload Classroom Video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
        with col_sample:
            st.markdown("<br>", unsafe_allow_html=True)
            use_sample = st.button("🎬 Use Pre-loaded Sample Video", help="Quickly test with the generated synthetic classroom video")

        active_video_path = None
        if uploaded_video:
            temp_vid = PROJECT_ROOT / "data" / f"temp_{uploaded_video.name}"
            with open(temp_vid, "wb") as f:
                f.write(uploaded_video.read())
            active_video_path = temp_vid
        elif use_sample and sample_video_path.exists():
            active_video_path = sample_video_path

        if active_video_path:
            st.info(f"Loaded Video: `{Path(active_video_path).name}`")
            
            if st.button("🚀 Run Automated Attendance Pipeline", type="primary"):
                session_id = f"SES-{datetime.now().strftime('%m%d%H%M')}"
                grade_val = sess_grade.split(" - ")[0]
                sec_val = sess_grade.split(" - ")[1].replace("Sec ", "")

                # Register session in DB
                attendance_svc.create_session(
                    session_id=session_id,
                    course_name=course_name,
                    instructor_name=instructor_name,
                    grade=grade_val,
                    section=sec_val,
                    video_source=str(active_video_path),
                )

                prog_bar = st.progress(0.0)
                status_txt = st.empty()

                def update_progress(val, msg):
                    prog_bar.progress(val)
                    status_txt.text(msg)

                # Configure services with thresholds
                services["fusion_engine"].min_presence_sec = min_duration
                services["fusion_engine"].face_confidence_thresh = conf_thresh
                services["vision_pipeline"].attendance_mgr.min_presence_duration_sec = min_duration
                services["vision_pipeline"].attendance_mgr.confidence_threshold = conf_thresh

                # Run automated attendance
                results = attendance_svc.process_automated_session(
                    session_id=session_id,
                    video_path=active_video_path,
                    audio_path=sample_audio_path if sample_audio_path.exists() else None,
                    grade=grade_val,
                    section=sec_val,
                    progress_callback=update_progress,
                )

                st.success("Automated processing finished! Review records below.")
                st.session_state["active_session_id"] = session_id

    with modality_tab2:
        st.subheader("Camera Snapshot Input")
        camera_img = st.camera_input("Capture classroom snapshot from connected camera")
        if camera_img:
            bytes_data = camera_img.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Run detection on snapshot
            roster = student_svc.get_all_students()
            annotated, tracks = services["vision_pipeline"].process_frame(
                frame=cv_img,
                timestamp_sec=0.0,
                student_roster=roster,
                is_detection_frame=True,
            )
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected Faces & Tracks", use_container_width=True)
            st.write(f"Faces identified: **{len(tracks)}**")

    # =========================================================================
    # Interactive Manual Review & Correction
    # =========================================================================
    active_sid = st.session_state.get("active_session_id")
    if active_sid:
        st.markdown("---")
        st.subheader(f"Attendance Verification Register: Session `{active_sid}`")
        st.caption("Review automated results, adjust status if necessary, and commit final attendance.")

        records = attendance_svc.get_session_attendance(active_sid)
        if records:
            review_rows = []
            for r in records:
                review_rows.append({
                    "Student ID": r["student_id"],
                    "Name": r.get("name", "Unknown"),
                    "AI Status": r["status"],
                    "Confidence %": round(r["confidence_score"] * 100, 1),
                    "Duration (s)": round(r["observation_duration_sec"], 1),
                    "Explanation": r.get("explanation", ""),
                })
            
            st.dataframe(pd.DataFrame(review_rows), use_container_width=True, hide_index=True)

            st.markdown("##### ✏️ Manual Teacher Correction / Override")
            c_sel, c_stat, c_note, c_btn = st.columns([1.5, 1, 2, 1])
            with c_sel:
                target_stu = st.selectbox("Student to Modify", [f"{r['name']} ({r['student_id']})" for r in records])
                target_sid = target_stu.split("(")[-1].replace(")", "").strip()
            with c_stat:
                new_status = st.selectbox("New Status", ["Present", "Late", "Absent", "Excused"])
            with c_note:
                override_note = st.text_input("Reason / Note", value="Teacher visual confirmation in class")
            with c_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Apply Override"):
                    ok = attendance_svc.manual_override_student(active_sid, target_sid, new_status, override_note)
                    if ok:
                        st.success(f"Updated {target_stu} to '{new_status}'!")
                        st.rerun()


# =============================================================================
# View 4: Audio Analysis
# =============================================================================

elif nav_choice == "🎙️ Audio Analysis":
    st.markdown('<div class="main-title">Classroom Acoustic Telemetry</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyze vocal activity, silence ratios, and aggregate discussion intensity</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ethical-banner">
        <strong>Privacy Principle:</strong> Acoustic telemetry is computed exclusively at the <em>aggregate classroom level</em>.
        No individual vocal biometrics are identified, and audio is never used as standalone proof to mark an individual absent or present.
    </div>
    """, unsafe_allow_html=True)

    col_up, col_btn = st.columns([2, 1])
    with col_up:
        uploaded_audio = st.file_uploader("Upload Classroom Audio (WAV, MP3, FLAC)", type=["wav", "mp3", "flac"])
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample_aud = st.button("🎵 Load Sample Classroom Audio", help="Load synthesized multi-phase classroom audio")

    target_audio_path = None
    sample_aud_path = PROJECT_ROOT / "data" / "sample_classroom.wav"

    if uploaded_audio:
        temp_aud = PROJECT_ROOT / "data" / f"temp_{uploaded_audio.name}"
        with open(temp_aud, "wb") as f:
            f.write(uploaded_audio.read())
        target_audio_path = temp_aud
    elif use_sample_aud and sample_aud_path.exists():
        target_audio_path = sample_aud_path

    if target_audio_path:
        st.audio(str(target_audio_path))
        with st.spinner("Analyzing classroom acoustic dynamics..."):
            aud_results = services["audio_processor"].analyze_audio(target_audio_path)

        # Display Key Acoustic Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Speech Ratio", f"{aud_results['speech_ratio'] * 100:.1f}%")
        with c2:
            st.metric("Silence Ratio", f"{aud_results['silence_ratio'] * 100:.1f}%")
        with c3:
            st.metric("Discussion Intensity", f"{aud_results['discussion_intensity_score'] * 100:.1f}%")
        with c4:
            st.metric("Speaker Turn Events", aud_results["turn_taking_events"])

        # Timeline plot of Voice Activity
        tbins = aud_results.get("temporal_bins", [])
        if tbins:
            st.subheader("Temporal Voice Activity & Energy Profile")
            df_aud = pd.DataFrame(tbins)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_aud["timestamp_sec"],
                y=df_aud["rms_energy"],
                name="RMS Acoustic Energy",
                line=dict(color="#2563EB", width=2),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.15)",
            ))
            fig.add_trace(go.Bar(
                x=df_aud["timestamp_sec"],
                y=[0.05 * v for v in df_aud["speech_active"]],
                name="Speech Activity (VAD)",
                marker_color="rgba(16, 185, 129, 0.4)",
            ))
            fig.update_layout(
                xaxis_title="Time (Seconds)",
                yaxis_title="Energy Magnitude",
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# View 5: Multimodal Fusion
# =============================================================================

elif nav_choice == "🔄 Multimodal Fusion":
    st.markdown('<div class="main-title">Multimodal Fusion & Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Synthesizing computer vision sightings with classroom acoustic context with transparent reasoning</div>', unsafe_allow_html=True)

    sessions = db.get_all_sessions()
    if not sessions:
        st.info("No recorded sessions found. Run a session in 'Take Attendance' first.")
    else:
        sess_map = {f"{s['course_name']} ({s['session_date']}) - {s['session_id']}": s["session_id"] for s in sessions}
        picked_label = st.selectbox("Select Session to Inspect Multimodal Evidence", list(sess_map.keys()))
        picked_sid = sess_map[picked_label]

        att_records = db.get_session_attendance(picked_sid)
        aud_m = db.get_audio_metrics(picked_sid)

        col_left, col_right = st.columns([1, 1.2])
        
        with col_left:
            st.subheader("Session Modality Summary")
            st.markdown(f"**Session ID:** `{picked_sid}`")
            if aud_m:
                st.markdown(f"**Classroom Speech Activity:** {aud_m['speech_ratio'] * 100:.1f}%")
                st.markdown(f"**Discussion Index:** {aud_m['discussion_intensity_score'] * 100:.1f}%")
                st.markdown(f"**Turn Transitions:** {aud_m['turn_taking_events']}")
            else:
                st.caption("Acoustic context not recorded for this session.")

            present_n = sum(1 for a in att_records if a["status"] == "Present")
            late_n = sum(1 for a in att_records if a["status"] == "Late")
            absent_n = sum(1 for a in att_records if a["status"] == "Absent")
            st.markdown(f"**Attendance Counts:** 🟢 {present_n} Present | 🟡 {late_n} Late | 🔴 {absent_n} Absent")

        with col_right:
            st.subheader("Transparent Explainability Rules")
            st.markdown("""
            - **Rule 1 (Vision Proof):** Continuous tracked observation duration must meet or exceed the session threshold (e.g. &ge; 5.0s).
            - **Rule 2 (Biometric Match):** Cosine similarity between face embedding and consented reference must meet threshold (&ge; 65%).
            - **Rule 3 (Consent Check):** Faces without consent are never correlated to student identities.
            - **Rule 4 (Acoustic Context):** Audio provides classroom engagement indicators but is never used as standalone proof of individual student presence.
            """)

        st.subheader("Student Explainability Cards")
        for r in att_records:
            status_color = "#16A34A" if r["status"] == "Present" else ("#D97706" if r["status"] == "Late" else "#DC2626")
            with st.expander(f"{r['name']} ({r['student_id']}) — Status: {r['status']}"):
                st.markdown(f"**Current Status:** <span style='color: {status_color}; font-weight: 700;'>{r['status']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Observed Duration:** `{r.get('observation_duration_sec', 0):.1f} seconds` ({r.get('frames_detected', 0)} frames)")
                st.markdown(f"**Facial Match Confidence:** `{r.get('confidence_score', 0) * 100:.1f}%`")
                st.markdown(f"**Marking Source:** `{r.get('marking_source', 'AI_AUTOMATED')}`")
                st.markdown(f"**AI Reasoning:** *{r.get('explanation', 'No explanation logged')}*")
                if r.get("manual_override_note"):
                    st.info(f"Manual Override Note: {r['manual_override_note']}")


# =============================================================================
# View 6: Engagement Analytics
# =============================================================================

elif nav_choice == "📈 Engagement Analytics":
    st.markdown('<div class="main-title">Classroom Engagement Telemetry & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Cautious, aggregate behavioral trends over time with ethical AI safeguards</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ethical-banner">
        <strong>⚠️ Crucial Ethical AI Boundary:</strong> These indicators represent <em>aggregate behavioral proxies</em> 
        (such as collective head pose stability, speech activity, and room motion). 
        The system <strong>does not claim</strong> to measure a student's actual internal thoughts, feelings, comprehension, or emotions.
    </div>
    """, unsafe_allow_html=True)

    sessions = db.get_all_sessions()
    if sessions:
        sess_map = {f"{s['course_name']} ({s['session_date']})": s["session_id"] for s in sessions}
        chosen_sess_lbl = st.selectbox("Select Session for Timeline Analytics", list(sess_map.keys()))
        chosen_sid = sess_map[chosen_sess_lbl]

        logs = db.get_session_engagement_logs(chosen_sid)
        if logs:
            df_logs = pd.DataFrame(logs)

            # Interactive Plotly Timeline
            st.subheader("Multimodal Engagement Timeline")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_logs["timestamp_sec"],
                y=[s * 100 for s in df_logs["aggregate_engagement_score"]],
                name="Engagement Index %",
                line=dict(color="#3B82F6", width=3),
            ))
            fig.add_trace(go.Scatter(
                x=df_logs["timestamp_sec"],
                y=[f * 10 for f in df_logs["faces_detected"]],
                name="Visible Faces (Scaled x10)",
                line=dict(color="#10B981", dash="dash"),
            ))
            fig.update_layout(
                xaxis_title="Timeline (Seconds into Session)",
                yaxis_title="Score / Metric Value",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Pedagogical Distribution Donut
            c_d1, c_d2 = st.columns([1, 1])
            with c_d1:
                st.subheader("Pedagogical Classroom States")
                state_counts = df_logs["pedagogical_state"].value_counts()
                fig_state = px.pie(
                    names=state_counts.index,
                    values=state_counts.values,
                    hole=0.5,
                    color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"],
                )
                fig_state.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_state, use_container_width=True)

            with c_d2:
                st.subheader("Teacher Advisory Insights")
                advisories = services["fusion_engine"].generate_pedagogical_advisory(logs)
                if advisories:
                    for adv in advisories:
                        st.info(f"**{adv['type']}:** {adv['message']}")
                else:
                    st.success("Session maintained steady participation patterns throughout.")
        else:
            st.info("No temporal engagement logs found for this session.")


# =============================================================================
# View 7: Reports & Export
# =============================================================================

elif nav_choice == "📑 Reports & Export":
    st.markdown('<div class="main-title">Attendance Reports & CSV Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Generate student attendance registers, summary percentages, and compliant data exports</div>', unsafe_allow_html=True)

    rep_tab1, rep_tab2, rep_tab3 = st.tabs(["📊 Student Summary Report", "📝 Session-wise Register", "🔒 De-identified Research Export"])

    with rep_tab1:
        st.subheader("Student Overall Attendance Summary")
        df_summary = reporting_svc.get_student_attendance_summary()
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        csv_summary = reporting_svc.export_student_summary_csv()
        st.download_button(
            label="📥 Download Student Summary (CSV)",
            data=csv_summary,
            file_name=f"student_attendance_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with rep_tab2:
        st.subheader("Session Register")
        sessions = db.get_all_sessions()
        if sessions:
            sess_map = {f"{s['course_name']} ({s['session_date']})": s["session_id"] for s in sessions}
            chosen_lbl = st.selectbox("Select Session to Export", list(sess_map.keys()))
            chosen_sid = sess_map[chosen_lbl]

            df_sess = reporting_svc.get_session_attendance_report(chosen_sid)
            st.dataframe(df_sess, use_container_width=True, hide_index=True)

            csv_sess = reporting_svc.export_session_report_csv(chosen_sid)
            st.download_button(
                label="📥 Download Session Register (CSV)",
                data=csv_sess,
                file_name=f"session_register_{chosen_sid}.csv",
                mime="text/csv",
            )

    with rep_tab3:
        st.subheader("De-identified Data Export for Research")
        st.markdown("""
        Produces an ethical research export where all student names and IDs are replaced 
        with cryptographic SHA-256 tokens (`Participant_abc123`).
        """)
        df_raw = reporting_svc.get_student_attendance_summary()
        df_anon = privacy_svc.anonymize_attendance_export(df_raw)
        st.dataframe(df_anon, use_container_width=True, hide_index=True)

        st.download_button(
            label="🔒 Download De-identified Dataset (CSV)",
            data=df_anon.to_csv(index=False),
            file_name="deidentified_classroom_research.csv",
            mime="text/csv",
        )


# =============================================================================
# View 8: Privacy & Settings
# =============================================================================

elif nav_choice == "⚙️ Privacy & Settings":
    st.markdown('<div class="main-title">Privacy, Retention & System Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Audit trail verification, automated retention purging, and compliance controls</div>', unsafe_allow_html=True)

    st.subheader("Data Retention Policy")
    col_ret1, col_ret2 = st.columns([2, 1])
    with col_ret1:
        ret_days = st.selectbox(
            "Configurable Retention Window for Session Logs & Footage",
            [30, 60, 90, 180, 365],
            index=2,
            help="Sessions and biometric observation logs older than this threshold will be permanently deleted.",
        )
    with col_ret2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Run Retention Purge Now", type="secondary"):
            deleted = privacy_svc.execute_retention_purge(retention_days=ret_days)
            st.success(f"Retention purge executed. Removed {deleted} expired sessions.")
            st.rerun()

    st.markdown("---")
    st.subheader("Privacy Audit Trail")
    st.caption("Immutable chronological record of biometric registrations, revocations, and teacher attendance overrides.")

    audit_df = privacy_svc.get_audit_trail_df(limit=100)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)
