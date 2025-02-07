# database.py
import sqlite3
import numpy as np  # Import numpy
import csv

class Database:
    def __init__(self, db_file='attendance.db'):
        """
        Initialize the database connection and create tables if they don't exist.
        :param db_file: Path to the SQLite database file.
        """
        self.conn = sqlite3.connect(db_file, check_same_thread=False) #For multithreading
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        """
        Create the necessary tables in the database.
        """
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                lrn TEXT UNIQUE NOT NULL,
                section TEXT NOT NULL,
                face_embedding BLOB NOT NULL
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lrn TEXT NOT NULL,
                status TEXT NOT NULL,
                temperature REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add_student(self, name, lrn, section, face_embedding):
        """
        Add a new student to the database.
        :param name: Student's name.
        :param lrn: Student's LRN (unique identifier).
        :param section: Student's section.
        :param face_embedding: Face embedding as a NumPy array (will be converted to bytes).
        """
        try:
            self.cursor.execute("""
                INSERT INTO students (name, lrn, section, face_embedding)
                VALUES (?, ?, ?, ?)
            """, (name, lrn, section, face_embedding.tobytes()))  # Convert to bytes
            self.conn.commit()
        except sqlite3.IntegrityError:
            print(f"Student with LRN {lrn} already exists.")


    def student_exists(self, lrn):
        """
        Check if a student with the given LRN exists.
        :param lrn: Student's LRN.
        :return: True if the student exists, False otherwise.
        """
        self.cursor.execute("SELECT COUNT(*) FROM students WHERE lrn = ?", (lrn,))
        return self.cursor.fetchone()[0] > 0

    def update_attendance(self, lrn, status, temperature=None):
        """
        Record a student's attendance.
        :param lrn: Student's LRN.
        :param status: Attendance status ('Present', 'Absent', etc.).
        :param temperature: Measured temperature (optional).
        """
        self.cursor.execute("""
            INSERT INTO attendance (lrn, status, temperature)
            VALUES (?, ?, ?)
        """, (lrn, status, temperature))
        self.conn.commit()



    def reset_attendance(self):
        """
        Reset all attendance records (delete all rows from the attendance table).
        """
        self.cursor.execute("DELETE FROM attendance")
        self.conn.commit()


    def delete_student(self, lrn):
        """
        Delete a student from the database (and their attendance records).
        :param lrn: Student's LRN.
        """
        # First, delete attendance records for the student
        self.cursor.execute("DELETE FROM attendance WHERE lrn = ?", (lrn,))
        # Then, delete the student from the students table
        self.cursor.execute("DELETE FROM students WHERE lrn = ?", (lrn,))
        self.conn.commit()



    def get_all_students(self):
        """
        Retrieve all students' face embeddings and LRNs.
        :return: Tuple of (known_face_embeddings, known_face_lrns).
        """
        self.cursor.execute("SELECT face_embedding, lrn FROM students")
        rows = self.cursor.fetchall()
        embeddings = [np.frombuffer(row[0], dtype=np.float64) for row in rows]  # Convert from bytes
        lrns = [row[1] for row in rows]
        return embeddings, lrns


    def get_all_students_with_attendance(self):
        """
        Retrieve all students and their *most recent* attendance status.
        :return: List of tuples (name, lrn, section, attendance_status, temperature).
        """
        self.cursor.execute("""
            SELECT s.name, s.lrn, s.section, 
                   COALESCE(a.status, 'Absent') AS status,  -- Use COALESCE for most recent
                   a.temperature
            FROM students s
            LEFT JOIN (
                SELECT lrn, status, temperature,
                       ROW_NUMBER() OVER (PARTITION BY lrn ORDER BY timestamp DESC) as rn
                FROM attendance
            ) a ON s.lrn = a.lrn AND a.rn = 1  -- Get only the most recent record
        """)
        return self.cursor.fetchall()


    def export_to_csv(self, file_path):
        """
        Export attendance records to a CSV file.
        :param file_path: Path to save the CSV file.
        """

        self.cursor.execute("""
            SELECT s.name, s.lrn, s.section, a.status, a.temperature, a.timestamp
            FROM attendance a
            INNER JOIN students s ON a.lrn = s.lrn
        """)
        rows = self.cursor.fetchall()

        with open(file_path, "w", newline='') as f:  # Use newline='' for CSV
            writer = csv.writer(f)
            writer.writerow(["Name", "LRN", "Section", "Status", "Temperature", "Timestamp"])  # Header
            writer.writerows(rows)  # Write all rows

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()