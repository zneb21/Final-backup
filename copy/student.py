# student.py
import numpy as np

class Student:
    def __init__(self, name, lrn, section, face_embedding=None):
        """
        Initialize a Student object.
        :param name: Student's name.
        :param lrn: Student's LRN (unique identifier).
        :param section: Student's section.
        :param face_embedding: Face embedding as a NumPy array (optional).
        """
        self.name = name
        self.lrn = lrn
        self.section = section
        self.face_embedding = face_embedding  # Store as NumPy array


    # Removed update_attendance and reset_attendance - database logic belongs in Database class

    # Removed to_dict and from_dict - not needed for direct SQLite interaction