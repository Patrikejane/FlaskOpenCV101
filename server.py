from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import face_recognition
import base64

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')  # Frontend interface

@socketio.on('video_frame')
def handle_video_frame(data):
    # Decode base64 frame
    frame_data = data.split(',')[1]
    frame = base64.b64decode(frame_data)
    nparr = np.frombuffer(frame, np.uint8)
    current_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize for processing
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

    # Detect faces
    all_face_locations = face_recognition.face_locations(
        current_frame_small, number_of_times_to_upsample=2, model='hog'
    )

    # Draw rectangles around faces
    for top, right, bottom, left in all_face_locations:
        top, right, bottom, left = top*4, right*4, bottom*4, left*4
        cv2.rectangle(current_frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Encode processed frame back to base64
    _, buffer = cv2.imencode('.jpg', current_frame)
    processed_frame = base64.b64encode(buffer).decode('utf-8')
    socketio.emit('processed_frame', {'image': 'data:image/jpeg;base64,' + processed_frame})

if __name__ == '__main__':
    socketio.run(app, debug=True)
