import serial
import time
import cv2
import numpy as np
from deepface import DeepFace
import re  # Import the regular expression module


class ArduinoCommunication:
    def __init__(self, port=None, baudrate=9600):  # Allow port to be None initially
        """
        Initialize the serial connection with the Arduino.
        :param port: The serial port where the Arduino is connected (e.g., 'COM3' on Windows).
        :param baudrate: The baud rate for serial communication (default is 9600).
        """
        self.ser = None  # Initialize ser to None
        self.port = port
        self.baudrate = baudrate
        if port: # Only try to connect if port is given
            self.connect()


    def connect(self):
        """
        Establish serial connection with the Arduino
        """
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=2)  # Add timeout
            time.sleep(2)  # Wait for Arduino initialization
            print(f"Successfully connected to Arduino on port {self.port}")
        except serial.SerialException as e:
            print(f"SerialException: Could not open port {self.port}: {e}")
            self.ser = None
        except Exception as e:
            print(f"Unexpected error opening serial port: {e}")
            self.ser = None

    def set_port(self, port):
        """
        Set (or change) the serial port. Closes existing connection if necessary.
        """
        if self.ser:
            self.close()
        self.port = port
        self.connect()  # Attempt to connect to the new port


    def get_temperature(self):
        """
        Read temperature data from the Arduino, handling non-numeric characters.
        Returns:
            float: The temperature in Celsius, or None if reading/parsing fails.
        """
        if self.ser is None:
            print("Arduino not connected.")
            return None

        try:
            self.ser.write(b'T')  # Send command
            time.sleep(0.1)
            data_str = self.ser.readline().decode('utf-8').strip()

            if not data_str:  # Handle empty string
                print("No data received from Arduino.")
                return None

            # Extract numeric part using regular expressions

            match = re.search(r"[-+]?\d*\.?\d+", data_str)  # Find number (with optional sign and decimal)
            if match:
                temp_str = match.group(0)  # Get the matched number string
                try:
                    temperature = float(temp_str)
                    return temperature
                except ValueError:
                    print(f"Failed to convert to float: {temp_str}")
                    return None
            else:
                print(f"No numeric data found in string: {data_str}")
                return None

        except serial.SerialException as e:
            print(f"Serial communication error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def close(self):
        """
        Close the serial connection.
        """
        if self.ser:
            self.ser.close()
            self.ser = None  # Set ser to None after closing
            print("Arduino connection closed.")


def capture_face_embedding(frame): # Take frame as input
    """
    Capture a face embedding from the provided frame.
    :param frame: The frame to process.
    :return: Face embedding if successful, otherwise None.
    """
    try:
        # 1. Detect faces
        face_locations = DeepFace.extract_faces(frame, detector_backend="opencv", enforce_detection=False)

        if not face_locations:
            print("No face detected in frame.")
            return None

        # 2. If a face is detected, then call represent.  We only take the *first* detected face.
        result = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)

        # Extract the embedding from the result
        if isinstance(result, list) and len(result) > 0:
            embedding = result[0]["embedding"]
        elif isinstance(result, dict) and "embedding" in result:
            embedding = result["embedding"]
        else:
            raise ValueError("Unexpected result format from DeepFace.represent.")

        return np.array(embedding)

    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None


def recognize_face(embedding, known_face_embeddings, known_face_lrns):
    """
    Recognize a face given its embedding.
    :param embedding: The pre-calculated face embedding.
    :param known_face_embeddings: List of known face embeddings.
    :param known_face_lrns: List of corresponding LRNs.
    :return: Tuple of (match_found, lrn, name).  match_found is True if a match
             is found within the threshold, False otherwise.  lrn and name
             are "Unknown" if no match is found, or if the input embedding is None.
    """
    if embedding is None:
        return False, "Unknown", "Unknown"  # No face detected

    try:
        # Compare the embedding with known embeddings using cosine distance
        distances = [cosine_distance(embedding, e) for e in known_face_embeddings]
        min_distance = min(distances)

        # Threshold for matching (cosine distance is between 0 and 2)
        if min_distance < 0.3:  # Adjust threshold as needed
            match_index = distances.index(min_distance)
            lrn = known_face_lrns[match_index]
            name = f"LRN: {lrn}"
            return True, lrn, name
        else:
            return False, "Unknown", "Unknown"  # No match found

    except Exception as e:
        print(f"Error recognizing face: {e}")
        return False, "Unknown", "Unknown"  # Error during recognition

def cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def cosine_distance(embedding1, embedding2):
    return 1 - cosine_similarity(embedding1, embedding2)