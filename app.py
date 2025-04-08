from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from activity_detector import detect_activity
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

video_path = None

def generate_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    with open('activity_log.txt', 'w') as log:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            try:
                result = detect_activity(frame)
                print(f"Frame {frame_count} result: {result}")
                status, debug_frame, wrist_y = result  # Now expecting 3 values
            except ValueError as e:
                print(f"ValueError: {e}")
                status = f"Error: Detection Failed ({str(e)})"
                debug_frame = frame
                wrist_y = None

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"Frame {frame_count} [{timestamp}]: {status}\n")
            log.flush()

            # Overlay status and wrist_y on frame
            cv2.putText(debug_frame, f"Status: {status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if wrist_y is not None:
                cv2.putText(debug_frame, f"Wrist Y: {wrist_y:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', debug_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            frame_count += 1
    cap.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    global video_path
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        if file:
            filename = "uploaded_video.mp4"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            print(f"Video saved at: {video_path}")
            return redirect(url_for('index'))
    return render_template('index.html', video_path=video_path)

@app.route('/video_feed')
def video_feed():
    global video_path
    print(f"Processing video at: {video_path}")
    if video_path and os.path.exists(video_path):
        return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "No video uploaded yet", 404

if __name__ == '__main__':
    app.run(debug=True)