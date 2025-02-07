import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import sqlite3
import cv2
import threading
import multiprocessing
import numpy as np
from deepface import DeepFace
import utils
import ast  # Import for safely evaluating strings


class Database:
    def __init__(self):
        self.conn = sqlite3.connect("attendance.db", check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                lrn TEXT UNIQUE NOT NULL,
                section TEXT NOT NULL,
                face_embedding BLOB NOT NULL
            )
        """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lrn TEXT NOT NULL,
                status TEXT NOT NULL,
                temperature REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.commit()

    def add_student(self, name, lrn, section, face_embedding):
        try:
            self.cursor.execute(
                """
                INSERT INTO students (name, lrn, section, face_embedding)
                VALUES (?, ?, ?, ?)
            """,
                (name, lrn, section, face_embedding.tobytes()),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError("Student with this LRN already exists.")

    def student_exists(self, lrn):
        self.cursor.execute("SELECT COUNT(*) FROM students WHERE lrn = ?", (lrn,))
        return self.cursor.fetchone()[0] > 0

    def get_all_students(self):
        self.cursor.execute("SELECT face_embedding, lrn FROM students")
        rows = self.cursor.fetchall()
        embeddings = [np.frombuffer(row[0], dtype=np.float64) for row in rows]
        lrns = [row[1] for row in rows]
        return embeddings, lrns

    def get_all_students_with_attendance(self):
        self.cursor.execute(
            """
            SELECT s.name, s.lrn, s.section, 
                   COALESCE(a.status, 'Absent') AS status,
                   a.temperature
            FROM students s
            LEFT JOIN (
                SELECT lrn, status, temperature,
                       ROW_NUMBER() OVER (PARTITION BY lrn ORDER BY timestamp DESC) as rn
                FROM attendance
            ) a ON s.lrn = a.lrn AND a.rn = 1
        """
        )
        return self.cursor.fetchall()

    def update_attendance(self, lrn, status, temperature=None):
        self.cursor.execute(
            """
            INSERT INTO attendance (lrn, status, temperature)
            VALUES (?, ?, ?)
        """,
            (lrn, status, temperature),
        )
        self.conn.commit()

    def reset_attendance(self):
        self.cursor.execute("DELETE FROM attendance")
        self.conn.commit()

    def delete_student(self, lrn):
        self.cursor.execute("DELETE FROM students WHERE lrn = ?", (lrn,))
        self.cursor.execute(
            "DELETE FROM attendance WHERE lrn=?", (lrn,)
        )  # Delete attendance too
        self.conn.commit()

    def export_to_csv(self, file_path):
        self.cursor.execute(
            """
            SELECT s.name, s.lrn, s.section, a.status, a.temperature, a.timestamp
            FROM attendance a
            INNER JOIN students s ON a.lrn = s.lrn
        """
        )
        rows = self.cursor.fetchall()

        with open(file_path, "w") as f:
            f.write("Name,LRN,Section,Status,Temperature,Timestamp\n")
            for row in rows:
                f.write(",".join(map(str, row)) + "\n")

    def close(self):
        self.conn.close()

    def get_student_by_lrn(self, lrn):
        """
        Retrieves a student's information by their LRN.

        Args:
            lrn: The LRN of the student to retrieve.

        Returns:
            A dictionary containing the student's information (name, lrn, section),
            or None if no student with the given LRN is found.
        """
        self.cursor.execute("SELECT name, lrn, section FROM students WHERE lrn = ?", (lrn,))
        row = self.cursor.fetchone()
        if row:
            # Create a dictionary to represent the student
            student_data = {
                'name': row[0],
                'lrn': row[1],
                'section': row[2]
            }
            return student_data
        else:
            return None  # Student not found

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Attendance System")
        self.root.geometry("800x600")
        self.db = Database()
        self.video_capture = None
        self.initialize_webcam()
        self.load_known_faces()
        self.create_gui()
        self.process_pool = multiprocessing.Pool(processes=4)  # Initialize process pool
        self.stop_flag = False
        self.frame_count = 0
        self.current_frame = None
        self.recognized_name = None  # Add this to store the name
        self.face_locations = None # Add this to store face location
        self.arduino = utils.ArduinoCommunication()
        self.arduino_port_entry = None

        # *** FIX: Call create_arduino_settings() here ***
        self.create_arduino_settings() # This was the missing piece!

    def initialize_webcam(self):
        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                raise Exception("Webcam not accessible.")
            print("Webcam initialized successfully.")
        except Exception as e:
            print(f"Error initializing webcam: {e}")
            messagebox.showerror("Error", "Failed to access webcam.")
            if hasattr(self, "capture_face_button"):
                self.capture_face_button.config(state=tk.DISABLED)
            if hasattr(self, "start_recognition_button"):
                self.start_recognition_button.config(state=tk.DISABLED)

    def load_known_faces(self):
        raw_embeddings, lrns = self.db.get_all_students()
        self.known_face_embeddings = raw_embeddings
        self.known_face_lrns = lrns
        print(f"Loaded {len(self.known_face_embeddings)} embeddings.")  # Database check
        # Add this print statement to verify embeddings are loaded
        if self.known_face_embeddings:
            print(
                "Sample embedding:", self.known_face_embeddings[0][:10]
            )  # Print first 10 elements of first embedding

    def create_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self.registration_tab = ttk.Frame(self.notebook)
        self.attendance_tab = ttk.Frame(self.notebook)
        self.student_management_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.registration_tab, text="Registration")
        self.notebook.add(self.attendance_tab, text="Attendance")
        self.notebook.add(self.student_management_tab, text="Management")
        self.create_registration_tab()
        self.create_attendance_tab()
        self.create_student_management_tab()
        self.create_arduino_settings() # Call it here


    def create_registration_tab(self):
        tk.Label(self.registration_tab, text="Name:").grid(
            row=0, column=0, padx=10, pady=5
        )
        self.name_entry = tk.Entry(self.registration_tab)
        self.name_entry.grid(row=0, column=1, padx=10, pady=5)
        tk.Label(self.registration_tab, text="LRN:").grid(
            row=1, column=0, padx=10, pady=5
        )
        self.lrn_entry = tk.Entry(self.registration_tab)
        self.lrn_entry.grid(row=1, column=1, padx=10, pady=5)
        tk.Label(self.registration_tab, text="Section:").grid(
            row=2, column=0, padx=10, pady=5
        )
        self.section_combobox = ttk.Combobox(
            self.registration_tab, values=["A", "B", "C", "D"]
        )
        self.section_combobox.grid(row=2, column=1, padx=10, pady=5)
        self.capture_face_button = tk.Button(
            self.registration_tab, text="Capture Face", command=self.start_capture_face
        )
        self.capture_face_button.grid(row=3, column=0, columnspan=2, pady=10)
        self.add_student_button = tk.Button(
            self.registration_tab, text="Add Student", command=self.add_student
        )
        self.add_student_button.grid(row=4, column=0, columnspan=2, pady=10)

    def start_capture_face(self):
        self.face_embeddings = []
        self.capture_count = 0
        threading.Thread(target=self.capture_multiple_faces).start()

    def capture_multiple_faces(self):
        # Initialize video capture here, but *release* it in a 'finally' block
        video_capture = None  # Initialize to None
        try:
            video_capture = cv2.VideoCapture(0)  # Moved inside the thread
            if not video_capture.isOpened():
                raise Exception("Failed to open webcam in capture thread")

            while self.capture_count < 5:
                ret, frame = video_capture.read()
                if not ret:
                    print("Failed to read frame.")
                    break  # Exit the loop if we can't read a frame
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                try:
                    face_locations = DeepFace.extract_faces(
                        small_frame, detector_backend="opencv", enforce_detection=False
                    )
                    if not face_locations:
                        print("No face detected.")
                        cv2.imshow("Capturing", small_frame)
                        cv2.waitKey(500)
                        continue
                    embedding = utils.capture_face_embedding(small_frame)
                    if embedding is not None:
                        self.face_embeddings.append(embedding)
                        self.capture_count += 1
                    for face_info in face_locations:
                        # Corrected bounding box drawing:
                        x = face_info['facial_area']['x']
                        y = face_info['facial_area']['y']
                        w = face_info['facial_area']['w']
                        h = face_info['facial_area']['h']
                        cv2.rectangle(
                            small_frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                        )
                    cv2.imshow("Capturing", small_frame)
                    cv2.waitKey(500)
                except Exception as e:
                    print(f"Error during capture: {e}")
        except Exception as e:
            print(f"Error in capture_multiple_faces: {e}")
            messagebox.showerror("Capture Error", str(e))  # Show error in messagebox
        finally:
            if video_capture is not None and video_capture.isOpened():
                video_capture.release()
            cv2.destroyAllWindows()

        if self.face_embeddings:
            self.current_face_embedding = np.mean(self.face_embeddings, axis=0)
            messagebox.showinfo("Success", "Face captured.")
        else:
            messagebox.showerror("Error", "No faces captured.")


    def add_student(self):
        name = self.name_entry.get()
        lrn = self.lrn_entry.get()
        section = self.section_combobox.get()
        if not hasattr(self, "current_face_embedding"):
            messagebox.showerror("Error", "Capture a face first.")
            return
        if self.db.student_exists(lrn):
            messagebox.showerror("Error", "LRN already exists.")
            return
        self.db.add_student(name, lrn, section, self.current_face_embedding)
        print(f"Added: {name} ({lrn})")
        self.load_known_faces()
        self.load_students()
        messagebox.showinfo("Success", "Student added.")

    def create_attendance_tab(self):
        self.terminal = scrolledtext.ScrolledText(
            self.attendance_tab, wrap=tk.WORD, height=10
        )
        self.terminal.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.terminal.insert(tk.END, "Output:\n")
        self.start_recognition_button = tk.Button(
            self.attendance_tab, text="Start", command=self.confirm_start_recognition
        )
        self.start_recognition_button.grid(row=1, column=0, padx=10, pady=10)
        self.stop_recognition_button = tk.Button(
            self.attendance_tab,
            text="Stop",
            command=self.stop_recognition,
            state=tk.DISABLED,
        )
        self.stop_recognition_button.grid(row=1, column=1, padx=10, pady=10)
        self.reset_attendance_button = tk.Button(
            self.attendance_tab, text="Reset", command=self.reset_attendance
        )
        self.reset_attendance_button.grid(row=2, column=0, padx=10, pady=10)
        self.export_csv_button = tk.Button(
            self.attendance_tab, text="Export", command=self.export_to_csv
        )
        self.export_csv_button.grid(row=2, column=1, padx=10, pady=10)
        # Modified Treeview columns
        self.tree = ttk.Treeview(
            self.attendance_tab,
            columns=("Name", "LRN", "Section", "Temperature", "Attendance"),  # Added "Temperature"
            show="headings",
        )
        self.tree.heading("Name", text="Name")
        self.tree.heading("LRN", text="LRN")
        self.tree.heading("Section", text="Section")
        self.tree.heading("Temperature", text="Temperature")  # Added Temperature heading
        self.tree.heading("Attendance", text="Attendance")
        self.tree.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
        self.load_students()

    def confirm_start_recognition(self):
        response = messagebox.askyesno("Confirm", "Start recognition?")
        if response:
            self.start_recognition()

    def start_recognition(self):
        print("start_recognition called!")
        if self.video_capture is None or not self.video_capture.isOpened():
            messagebox.showerror("Error", "Webcam not accessible.")
            return
        print("Starting recognition...")
        self.stop_flag = False
        self.start_recognition_button.config(state=tk.DISABLED)
        self.stop_recognition_button.config(state=tk.NORMAL)
        self.run_recognition()
        print("run_recognition called")

    def stop_recognition(self):
        print("Stopping recognition...")
        self.stop_flag = True
        self.start_recognition_button.config(state=tk.NORMAL)
        self.stop_recognition_button.config(state=tk.DISABLED)

    def run_recognition(self):
        if self.stop_flag:
            print("run_recognition: stop_flag is True, returning")
            return
        if self.video_capture is None or not self.video_capture.isOpened():
            print("run_recognition: Webcam not accessible.")
            self.stop_recognition()
            return
        ret, frame = self.video_capture.read()
        if not ret:
            print("run_recognition: Failed to read frame.")
            self.stop_recognition()
            return
        self.frame_count += 1
        if self.frame_count % 5 != 0:
            print(f"run_recognition: Skipping frame {self.frame_count}")
            self.root.after(10, self.run_recognition)
            return
        print(f"run_recognition: Processing frame {self.frame_count}")
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        self.current_frame = small_frame
        self.root.after(0, self.display_frame)

    def display_frame(self):
        if self.current_frame is not None:
            frame = self.current_frame.copy()  # Work on a copy

            # Draw bounding box (if face locations are available)
            if self.face_locations:
                for face_info in self.face_locations:
                    x = face_info['facial_area']['x']
                    y = face_info['facial_area']['y']
                    w = face_info['facial_area']['w']
                    h = face_info['facial_area']['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Draw name (if available) *above* the bounding box
                    if self.recognized_name:
                        text_y = y - 10  # Position text above the box
                        if text_y < 10:  # Make sure text is within frame bounds
                            text_y = y + h + 20 # If y -10 is offscreen, put text below
                        cv2.putText(
                            frame,
                            self.recognized_name,  # Display the NAME
                            (x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

            cv2.imshow("Attendance", frame)
            cv2.waitKey(1)

        if not self.stop_flag:
            self.process_next_frame()


    def process_next_frame(self):
        if self.current_frame is None:
            print("process_next_frame: No frame.")
            return

        try:
            # 1. Extract face embedding from the current frame *in the main thread*
            embedding = utils.capture_face_embedding(self.current_frame.copy())

            if embedding is None:
                print("process_next_frame: No face detected in frame.")
                self.recognized_name = None  # Reset name
                self.face_locations = [] # Reset the location as well.
                if not self.stop_flag:
                    self.root.after(10, self.run_recognition)
                return

            # 2. Use apply_async to call recognize_face (no longer DeepFace.find!)
            result = self.process_pool.apply_async(
                utils.recognize_face,
                args=(embedding, self.known_face_embeddings, self.known_face_lrns),
            )
            self.root.after(0, self.handle_recognition_result, result)  # Pass AsyncResult

        except Exception as e:
            print(f"Error in processing: {e}")
            if not self.stop_flag:
                self.root.after(10, self.run_recognition)

    def handle_recognition_result(self, result):
        try:
            # Get the result from recognize_face (match_found, lrn, name)
            match_found, lrn, _ = result.get(timeout=1)  # We don't need the 'name' from here anymore

            if match_found:
                # Retrieve student information from the database
                student_data = self.db.get_student_by_lrn(lrn)
                if student_data:
                    self.recognized_name = student_data['name']  # Store the student's NAME
                    self.safe_log(f"Recognized: {self.recognized_name}")
                else:
                    self.recognized_name = "Unknown (LRN Found)" #Should never reach here if database is working correctly
                    self.safe_log(f"Recognized LRN {lrn} but student data not found!")


                # --- Get face locations for bounding box ---
                try:
                    self.face_locations = DeepFace.extract_faces(  # Store locations
                        self.current_frame,
                        detector_backend="opencv",
                        enforce_detection=False,
                    )

                except Exception as e:
                    print(f"Error drawing bounding box: {e}")
                    self.face_locations = None  # Set to None on error

                temperature = self.arduino.get_temperature()
                if temperature is not None:
                    self.safe_log(f"  Temp: {temperature:.2f}°C")
                    self.db.update_attendance(lrn, "Present", temperature)
                    self.load_students() # Reload data into the Treeview
                else:
                    self.safe_log("  Failed to read temp.")
            else:
                self.safe_log("Unknown person.")
                self.recognized_name = "Unknown"  # Set name to "Unknown"
                 # --- Get face locations for bounding box ---
                try:
                    self.face_locations = DeepFace.extract_faces(  # Store locations
                        self.current_frame,
                        detector_backend="opencv",
                        enforce_detection=False,
                    )

                except Exception as e:
                    print(f"Error drawing bounding box: {e}")
                    self.face_locations = None  # Set to None on error

        except multiprocessing.TimeoutError:
            print("Timeout.")
        except Exception as e:
            print(f"Error in handle_recognition_result: {e}")

        if not self.stop_flag:
            self.root.after(10, self.run_recognition)

    def reset_attendance(self):
        self.db.reset_attendance()
        self.load_students()
        print("Attendance reset")

    def export_to_csv(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")]
        )
        if file_path:
            self.db.export_to_csv(file_path)
            messagebox.showinfo("Success", "Exported.")

    def delete_student(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showerror("Error", "Select a student.")
            return
        lrn = self.tree.item(selected_item, "values")[1]
        self.db.delete_student(lrn)
        self.load_students()

    def load_students(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        students = self.db.get_all_students_with_attendance()
        for student in students:
            # student data: (name, lrn, section, attendance_status, temperature)
            name = student[0]
            lrn = student[1]
            section = student[2]
            status = student[3]
            # Handle potential None for temperature:
            temperature = student[4] if student[4] is not None else "N/A"  # Placeholder
            self.tree.insert("", "end", values=(name, lrn, section, temperature, status)) # Added temperature

    def create_student_management_tab(self):
        tk.Label(self.student_management_tab, text="Manage Students").pack(pady=20)
        self.delete_student_button = tk.Button(
            self.student_management_tab, text="Delete Student", command=self.delete_student
        )
        self.delete_student_button.pack(pady=10)

    def create_arduino_settings(self):
        arduino_frame = ttk.LabelFrame(self.root, text="Arduino")
        arduino_frame.pack(padx=10, pady=10)
        tk.Label(arduino_frame, text="Port:").grid(row=0, column=0, padx=5, pady=5)
        self.arduino_port_entry = tk.Entry(arduino_frame) # Created the entry!
        self.arduino_port_entry.grid(row=0, column=1, padx=5, pady=5)
        self.arduino_port_entry.insert(0, "COM11") # Put default here.
        connect_button = tk.Button(
            arduino_frame, text="Connect", command=self.connect_arduino
        )
        connect_button.grid(row=0, column=2, padx=5, pady=5)

    def connect_arduino(self):
        port = self.arduino_port_entry.get() # Now it should work!
        self.arduino.set_port(port)
        if self.arduino.ser is not None:
            messagebox.showinfo("Arduino", "Connected!")
        else:
            messagebox.showerror("Arduino", "Failed.")

    def safe_log(self, message):
        if hasattr(self, "terminal") and self.terminal:
            self.terminal.insert(tk.END, f"{message}\n")
            self.terminal.see(tk.END)
        else:
            print(message)

    def on_closing(self):
        if self.video_capture is not None and self.video_capture.isOpened():
            self.video_capture.release()
        self.process_pool.close()
        self.process_pool.join()
        self.db.close()
        self.arduino.close()
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    print("Before mainloop")
    root.mainloop()
    print("After mainloop")