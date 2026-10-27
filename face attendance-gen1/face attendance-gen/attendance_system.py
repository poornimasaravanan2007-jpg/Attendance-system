import os
import sqlite3
import datetime
import logging
import pickle
import cv2
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib

# CONFIGURATION
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# File paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "attendance_data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "attendance.db")
DATASET_DIR = os.path.join(DATA_DIR, "faces")
MODEL_PATH = os.path.join(DATA_DIR, "svm_model.joblib")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "label_map.pkl")

# Initialize face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# DATABASE FUNCTIONS
def migrate_db():
    """
    Ensure the 'students' table has the required columns.
    Adds missing columns using ALTER TABLE so existing data is preserved.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # If the students table does not exist, nothing to migrate here
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
        if cursor.fetchone() is None:
            conn.close()
            return

        # Get existing columns
        cursor.execute("PRAGMA table_info(students)")
        existing_cols = [row[1] for row in cursor.fetchall()]

        # columns we expect in the current schema with SQLite type definitions
        expected = {
            "reg_no": "TEXT",
            "class": "TEXT",
            "section": "TEXT",
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        }

        added = []
        for col, col_def in expected.items():
            if col not in existing_cols:
                # SQLite supports: ALTER TABLE table ADD COLUMN column_def
                try:
                    cursor.execute(f"ALTER TABLE students ADD COLUMN {col} {col_def}")
                    added.append(col)
                except Exception as e:
                    logger.error(f"Failed to add column {col}: {e}")

        if added:
            conn.commit()
            logger.info(f"DB migration: added columns to students table: {added}")
        conn.close()
    except Exception as e:
        logger.error(f"DB migration failed: {e}")

def init_db():
    """Initialize database and run migrations if needed"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Students table (won't modify existing table)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT,
            reg_no TEXT,
            class TEXT,
            section TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            student_name TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')

    # Create indices
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_timestamp ON attendance(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id)')

    conn.commit()
    conn.close()

    # run migration to add any missing columns to existing students table
    migrate_db()

    logger.info("Database initialized")

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# FACE PROCESSING FUNCTIONS
def preprocess_face(img, size=(64, 64), return_flat=True):
    """Extract and preprocess face for SVM (returns flattened vector by default)."""
    try:
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None

        # Take largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]

        # Resize and normalize
        face_resized = cv2.resize(face, size)
        face_normalized = face_resized.astype('float32') / 255.0

        if return_flat:
            # Flatten to 1D vector
            return face_normalized.flatten()
        else:
            # Keep as (H,W,1) if needed elsewhere
            return np.expand_dims(face_normalized, axis=-1)
    except Exception as e:
        logger.error(f"Error preprocessing face: {e}")
        return None

def save_model_data(model_pipeline, label_map):
    """Save sklearn pipeline and label_map (joblib/pickle)."""
    try:
        joblib.dump(model_pipeline, MODEL_PATH)
        with open(LABEL_MAP_PATH, 'wb') as f:
            pickle.dump(label_map, f)
        logger.info("SVM model and label map saved.")
    except Exception as e:
        logger.error(f"Error saving model data: {e}")

def load_model_data():
    """Load sklearn pipeline and label_map."""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
            model_pipeline = joblib.load(MODEL_PATH)
            with open(LABEL_MAP_PATH, 'rb') as f:
                label_map = pickle.load(f)
            return model_pipeline, label_map
    except Exception as e:
        logger.error(f"Error loading model: {e}")
    return None, None

def train_model():
    """Train an SVM pipeline on flattened face images."""
    try:
        X = []
        y = []

        if not os.path.exists(DATASET_DIR):
            return False, "No face data found"

        # Use sorted order so mapping is deterministic
        student_dirs = sorted([d for d in os.listdir(DATASET_DIR)
                               if os.path.isdir(os.path.join(DATASET_DIR, d))])

        if not student_dirs:
            return False, "No students added yet"

        # Map student IDs (folder names) to continuous indices for classifier
        label_map = {idx: int(sid) for idx, sid in enumerate(student_dirs)}
        inv_label_map = {int(sid): idx for idx, sid in label_map.items()}

        print("Processing images... this may take a moment.")
        for student_id in student_dirs:
            student_path = os.path.join(DATASET_DIR, student_id)
            image_files = [f for f in os.listdir(student_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_file in image_files:
                img_path = os.path.join(student_path, img_file)
                img = cv2.imread(img_path)
                processed = preprocess_face(img, return_flat=True)
                if processed is not None:
                    X.append(processed)
                    y.append(inv_label_map[int(student_id)])

        if len(X) < 10:
            return False, f"Not enough face data. Found {len(X)} samples."

        X = np.array(X)
        y = np.array(y)

        # Count distinct classes in the collected labels
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)

        if num_classes < 2:
            # Helpful message: SVM requires >=2 classes
            return False, (
                f"Need at least 2 different students (classes) to train SVM. "
                f"Found {num_classes}. Please add more students (with images) before training."
            )

        print(f"Training SVM with {len(X)} samples from {num_classes} students...")

        # Decide PCA components: at least 1, at most 100 or samples-1
        n_components = max(1, min(len(X) - 1, 100))

        steps = [
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced'))
        ]
        pipeline = Pipeline(steps)

        pipeline.fit(X, y)

        # Save pipeline and label map
        save_model_data(pipeline, label_map)

        # Evaluate on training set
        train_acc = pipeline.score(X, y)
        return True, f"SVM trained with {train_acc:.1%} training accuracy."

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False, f"Training failed: {str(e)}"

# CLI INTERFACE FUNCTIONS
def clean_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def add_student():
    print("\n--- Add New Student ---")
    name = input("Enter Full Name: ").strip()
    if not name:
        print("Error: Name is required!")
        return

    roll_no = input("Enter Roll Number (optional): ").strip()
    reg_no = input("Enter Registration Number (optional): ").strip()
    class_name = input("Enter Class (optional): ").strip()
    section = input("Enter Section (optional): ").strip()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO students (name, roll_no, reg_no, class, section) VALUES (?, ?, ?, ?, ?)",
        (name, roll_no, reg_no, class_name, section)
    )
    student_id = cursor.lastrowid
    conn.commit()
    conn.close()

    student_path = os.path.join(DATASET_DIR, str(student_id))
    os.makedirs(student_path, exist_ok=True)

    print(f"\nStudent {name} added with ID: {student_id}")
    input("Press Enter to start face capture (Hold Esc or q to quit capture early)...")

    cap = cv2.VideoCapture(0)
    count = 0
    total = 50

    try:
        while count < total:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera.")
                break

            # Display instructions on frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Capturing: {count}/{total}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Move your head slightly", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Detect face for visual feedback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Face Enrollment", display_frame)

            # Capture face every time a face is detected (up to total)
            if len(faces) > 0:
                count += 1
                img_name = os.path.join(student_path, f"face_{count}.jpg")
                cv2.imwrite(img_name, frame)

            # Check for Esc or q key, and also if window was closed
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

            # Additional check for window being closed via 'X' button
            if cv2.getWindowProperty("Face Enrollment", cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        print(f"\nFace Capture Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nCaptured {count} face images.")

    input("Press Enter to continue...")

def run_attendance():
    model_pipeline, label_map = load_model_data()
    if model_pipeline is None:
        print("Error: Model not trained yet! Please train the model first.")
        input("Press Enter to continue...")
        return

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM students")
    students = {row['id']: row['name'] for row in cursor.fetchall()}
    conn.close()

    if not students:
        print("Error: No students registered!")
        input("Press Enter to continue...")
        return

    print("\n--- Live Attendance Scanner ---")
    print("Hold 'Esc' or 'q' to exit scanner.")

    cap = cv2.VideoCapture(0)
    last_attendance = {}  # To avoid multiple logs for same person in short time
    COOLDOWN_SECONDS = 60

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera.")
                break

            display_frame = frame.copy()
            processed = preprocess_face(frame, return_flat=True)

            if processed is not None:
                try:
                    # Get probabilities for each class
                    probs = model_pipeline.predict_proba([processed])[0]
                    pred_idx = int(model_pipeline.predict([processed])[0])
                    confidence = float(probs[pred_idx] * 100.0)

                    # Map index back to student_id
                    student_id = label_map[pred_idx]

                    if confidence > 50:  # Confidence threshold
                        name = students.get(student_id, "Unknown")

                        # Check cooldown
                        now = datetime.datetime.now()
                        if student_id not in last_attendance or (now - last_attendance[student_id]).total_seconds() > COOLDOWN_SECONDS:
                            # Log attendance
                            conn = get_db()
                            cursor = conn.cursor()
                            cursor.execute(
                                "INSERT INTO attendance (student_id, student_name, confidence) VALUES (?, ?, ?)",
                                (student_id, name, float(confidence))
                            )
                            conn.commit()
                            conn.close()
                            last_attendance[student_id] = now
                            print(f"Logged: {name} ({confidence:.1f}%)")

                        # Visual feedback
                        cv2.putText(display_frame, f"{name} ({confidence:.1f}%)", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "Searching...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                except Exception as e:
                    logger.error(f"Prediction error: {e}")

            cv2.imshow("Attendance Scanner", display_frame)

            # Check for Esc or q key, and also if window was closed
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

            if cv2.getWindowProperty("Attendance Scanner", cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        print(f"\nScanner Error: {e}")
        input("Press Enter to continue...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def show_records():
    print("\n--- Attendance Records ---")
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT student_name, timestamp, confidence FROM attendance ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No records found.")
    else:
        # Print header
        print(f"\n{'Student Name':<20} | {'Timestamp':<25} | {'Confidence':<10}")
        print("-" * 65)
        for row in rows:
            # Safely handle None values for timestamp and confidence
            timestamp_val = row['timestamp'] if row['timestamp'] is not None else ""
            confidence_val = row['confidence']
            if confidence_val is None:
                conf_str = "N/A"
            else:
                try:
                    conf_str = f"{float(confidence_val):.2f}%"
                except Exception:
                    conf_str = "N/A"

            student_name = row['student_name'] if row['student_name'] is not None else "Unknown"
            print(f"{student_name:<20} | {timestamp_val:<25} | {conf_str:<10}")

    input("\nPress Enter to continue...")

def export_records():
    print("\n--- Export Attendance Records ---")
    conn = get_db()
    query = "SELECT student_name, timestamp, confidence FROM attendance ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No records to export.")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_report_{timestamp}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"Successfully exported {len(df)} records to: {filepath}")

    input("\nPress Enter to continue...")

def delete_student():
    print("\n--- Delete Student Data ---")
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, roll_no FROM students")
    students = cursor.fetchall()

    if not students:
        print("No students found in database.")
        conn.close()
        input("\nPress Enter to continue...")
        return

    print(f"\n{'ID':<5} | {'Name':<20} | {'Roll No':<15}")
    print("-" * 45)
    for s in students:
        print(f"{s['id']:<5} | {s['name']:<20} | {s['roll_no'] or 'N/A':<15}")

    sid_input = input("\nEnter Student ID to delete (or press Enter to cancel): ").strip()
    if not sid_input:
        conn.close()
        return

    try:
        sid = int(sid_input)
        # Check if exists
        cursor.execute("SELECT name FROM students WHERE id = ?", (sid,))
        student = cursor.fetchone()
        if not student:
            print("Error: Student ID not found.")
            conn.close()
            input("\nPress Enter to continue...")
            return

        confirm = input(f"Are you sure you want to delete {student['name']}? (y/n): ").lower()
        if confirm == 'y':
            # 1. Delete Attendance
            cursor.execute("DELETE FROM attendance WHERE student_id = ?", (sid,))
            # 2. Delete Student record
            cursor.execute("DELETE FROM students WHERE id = ?", (sid,))

            # 3. Delete face images
            import shutil
            student_path = os.path.join(DATASET_DIR, str(sid))
            if os.path.exists(student_path):
                shutil.rmtree(student_path)
                print(f"Deleted face dataset for {student['name']}.")

            conn.commit()
            print(f"Student {student['name']} and their records deleted successfully.")

            # Suggest retraining
            print("\nNOTE: You should retrain the AI model now that dataset has changed.")
    except Exception as e:
        print(f"Error during deletion: {e}")

    conn.close()
    input("\nPress Enter to continue...")

def main():
    init_db()
    while True:
        clean_screen()
        print("==============================")
        print("   AI ATTENDANCE SYSTEM CLI   ")
        print("==============================")
        print("1. Add New Student")
        print("2. Train AI Model")
        print("3. Run Attendance Scanner")
        print("4. View Attendance Records")
        print("5. Export Records to CSV")
        print("6. Delete Student")
        print("7. Exit")

        choice = input("\nSelect an option (1-7): ").strip()

        if choice == '1':
            add_student()
        elif choice == '2':
            success, msg = train_model()
            print(f"\n{msg}")
            input("Press Enter to continue...")
        elif choice == '3':
            run_attendance()
        elif choice == '4':
            show_records()
        elif choice == '5':
            export_records()
        elif choice == '6':
            delete_student()
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice, try again.")
            time.sleep(1)

if __name__ == '__main__':
    main()