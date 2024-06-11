from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import subprocess
import json
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('HandTalk.html')

@app.route('/process_sign_language', methods=['POST'])
def process_sign_language():
    try:
        # Get the image data from the POST request
        data = request.json
        image_data = data.get('image', None)

        if image_data:
            # Decode base64 image data
            image_bytes = base64.b64decode(image_data)

            # Save the image to a file
            with open('captured_frame.jpg', 'wb') as f:
                f.write(image_bytes)

            # Call the test.py script to process the frame
            result = subprocess.check_output(['python', 'test.py'])

            # Assuming test.py outputs the result as JSON
            result_json = json.loads(result)

            # Return the processed output as JSON
            return jsonify({'output': result_json['output']}), 200
        else:
            return jsonify({'output': 'Error: No image data provided.'}), 400
    except subprocess.CalledProcessError as e:
        # Handle errors if the subprocess (test.py) encounters an issue
        error_message = f'Error: {e.output.strip()}' if e.output else 'Unknown error'
        return jsonify({'output': error_message}), 500
    except Exception as e:
        # Log the exception traceback for further investigation
        print(f"Error processing sign language: {str(e)}")
        return jsonify({'output': 'Internal server error.'}), 500

@socketio.on('test_message')
def handle_test_message(data):
    # Get the test message data
    image_data = data.get('message', None)

    if image_data:
        # Save the image to a file (you can enhance this part)
        with open('captured_frame.jpg', 'wb') as f:
            f.write(image_data)

        try:
            # Call the test.py script to process the frame
            result = subprocess.check_output(['python', 'test.py'])

            # Assuming test.py outputs the result as JSON
            result_json = json.loads(result)

            # Emit the result back to the client via WebSocket
            emit('response', {'data': {'output': result_json['output'], 'running': False}})
        except subprocess.CalledProcessError as e:
            # Handle errors if the subprocess (test.py) encounters an issue
            error_message = f'Error: {e.output.strip()}' if e.output else 'Unknown error'
            emit('response', {'data': {'output': error_message, 'running': False}})
            print(f"Error executing test.py: {error_message}")

if __name__ == '__main__':
    # Use socketio.run instead of app.run
    socketio.run(app, debug=True)
