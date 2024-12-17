from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import face_recognition
import base64
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

# Helper function to decode base64 frame to OpenCV image
def decode_frame(frame_data):
    try:
        # Split the base64 data and decode
        frame_data = frame_data.split(',')[1]
        frame = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding frame: {e}")
        return None

# Helper function to detect faces and draw rectangles
def process_frame(frame):
    try:
        # Resize the frame for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Detect face locations
        face_locations = face_recognition.face_locations(
            small_frame, number_of_times_to_upsample=2, model='hog'
        )

        # Draw rectangles around faces on the original frame
        for top, right, bottom, left in face_locations:
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        return frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

# Helper function to encode frame to base64
def encode_frame_to_base64(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding frame: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')  # Frontend interface

@socketio.on('video_frame')
def handle_video_frame(data):
    # Decode the incoming frame
    current_frame = decode_frame(data)
    if current_frame is None:
        return  # Skip processing if frame could not be decoded

    # Process the frame (detect faces)
    processed_frame = process_frame(current_frame)
    if processed_frame is None:
        return  # Skip processing if frame could not be processed

    # Encode the processed frame back to base64
    processed_frame_base64 = encode_frame_to_base64(processed_frame)
    if processed_frame_base64 is None:
        return  # Skip sending the frame if encoding failed

    # Send the processed frame back to the client
    socketio.emit('processed_frame', {'image': f'data:image/jpeg;base64,{processed_frame_base64}'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
