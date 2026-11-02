"""
Database Initialization Script for Multimodal Attendance & Engagement Agent.
Run from command line:
    python database/db_init.py [--with-sample-data]
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_manager import DatabaseManager


def initialize_database(with_sample_data: bool = False) -> None:
    print("=" * 60)
    print("Initializing Multimodal Attendance & Engagement Database...")
    print("=" * 60)

    db = DatabaseManager()
    print(f"[OK] Database initialized successfully at: {db.db_path}")

    if with_sample_data:
        print("[INFO] Populating fictional sample data...")
        from sample_data.seed_database import seed_all
        seed_all(db)
        print("[OK] Sample data successfully populated.")

    stats = db.get_dashboard_stats()
    print("-" * 60)
    print(f"Total Registered Students: {stats['total_students']}")
    print(f"Consented Students:        {stats['consented_students']}")
    print(f"Classes / Sessions:        {stats['total_sessions']}")
    print(f"Overall Attendance:        {stats['overall_attendance_pct']}%")
    print("=" * 60)
    print("Database is ready for use.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize SQLite database.")
    parser.add_argument("--with-sample-data", action="store_true", help="Populate fictional sample data")
    args = parser.parse_args()
    initialize_database(with_sample_data=args.with_sample_data)
