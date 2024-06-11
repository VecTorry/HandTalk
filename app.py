import streamlit as st
import json
import websocket
import socketio
import cv2
import numpy as np
import base64

# Initialize variables
cameraOpened = False

# Set up the Streamlit app
st.title("Sign Language Interpreter")

# Initialize the camera state
camera_opened = st.sidebar.checkbox("Open Camera")

# Create a Socket.IO client
socket = socketio.Client()

# Function to open the camera
def open_camera(label):
    global cameraOpened
    print("Opening camera...")
    if not cameraOpened:
        try:
            # Use the label argument for the camera input
            frame = st.camera_input(label=label, use_container_width=True)

            # Convert the image data to a base64-encoded string
            image_data = canvas_to_base64(frame)

            # Send the image data to the Flask server via WebSocket
            socket.emit('test_message', {'message': 'test_image_data', 'image_data': image_data})

        except Exception as error:
            print('Error accessing the camera:', error)

# Function to send test message to test.py via WebSocket
def send_test_message():
    ws = websocket.WebSocket()
    ws.connect("ws://localhost:8501/test_ws")  # Adjust the WebSocket URL as needed
    ws.send(json.dumps({'message': 'test_image_data'}))
    response = ws.recv()
    ws.close()
    return response

# Open the camera if the checkbox is selected
if camera_opened:
    st.text("Capturing and processing frame...")

    # Capture frame
    frame = st.camera_input()

    # Convert the image data to a base64-encoded string
    _, buffer = cv2.imencode('.jpg', frame)
    image_data = base64.b64encode(buffer).decode('utf-8')

    # Send a test message to test.py via WebSocket
    test_message_response = send_test_message()

    # Display the captured frame and the test message response
    st.image(frame, caption="Captured Frame", use_column_width=True)
    st.text(f"Test Message Response: {test_message_response}")

# Define the canvas_to_base64 function
def canvas_to_base64(canvas_data):
    # Implement the function as needed
    pass

# Connect to the WebSocket server
socket.connect("http://localhost:8501")  # Replace with your WebSocket server URL

# Set the Streamlit app close handler
st.experimental_set_query_params(socket_id="test")
st.experimental_rerun()
st.title("Sign Language Interpreter")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("WebSocket connection established.")
    open_camera(label="Camera Feed")
