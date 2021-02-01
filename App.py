import json
from flask import Flask, request, Response, jsonify
from video_process import VideoCamera

app = Flask(__name__)


def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            print("video or ipcam not connected")
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video_feed')
def video_feed():
    name = request.args.get('name')
    video_process = VideoCamera(video='./videos/' + str(name))
    return Response(gen(video_process),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
