import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import json
import requests
import websocket
import asyncio

# Initialize coordinate variables for hand tracking
coordUpdated1 = [0, 1]
coordUpdated2 = [0, 1]
coordPre1 = [0, 1]
coordPre2 = [0, 1]

# Set parameters for hand tracking and gesture recognition
offset = 20
imgSize = 300
join_distance_threshold = 300  # Adjust the threshold as needed

# Specify the data folder and initialize a counter
folder = "Data/C"
counter = 0

# Update the labels with your own gesture labels for one-handed and two-handed gestures
labels_one_handed = ["I feel hot", "I need drink",
                     "I need food", "I need to use the toilet", "I'm Fine", "I'm sorry", "No",
                     "Okay", "Yes", "Hello/Bye", "What"]

labels_two_handed = ["Call the doctor", "I feel cold", "I feel sick", "I'm out of breath", "It hurts", "Later"
                     ,"Maybe", "Now", "Take care", "Thank you"]

# Initialize the webcam capture and hand detection modules
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier_one_handed = Classifier("Model/keras_model_oneHanded.h5", "Model/labels_oneHanded.txt")
classifier_two_handed = Classifier("Model/keras_model_twoHanded.h5", "Model/labels_twoHanded.txt")

# Initialize variables for tracking one-handed gesture statistics
total_frames_one_handed = 0
correct_predictions_one_handed = [0] * len(labels_one_handed)
confidence_one_handed = [0.0] * len(labels_one_handed)

# Initialize variables for tracking two-handed gesture statistics
total_frames_two_handed = 0
correct_predictions_two_handed = [0] * len(labels_two_handed)
confidence_two_handed = [0.0] * len(labels_two_handed)

# Additional variables for cumulative accuracy
cumulative_accuracy_one_handed = 0
cumulative_accuracy_two_handed = 0
total_frames = 0

# Import the necessary libraries for server communication
server_url = "http://127.0.0.1:5000/process_sign_language"  # Change the URL if your server is running on a different address
headers = {'Content-Type': 'application/json'}

# WebSocket server endpoint for handling test messages
async def test_ws_handler(websocket, path):
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get('message') == 'test_image_data':
                # Process the test message (replace this with your logic)
                print("Received test message:", data)
                response_data = {'status': 'success', 'message': 'Test message received successfully'}
                await websocket.send(json.dumps(response_data))
    except websocket.exceptions.ConnectionClosedError:
        print("WebSocket connection closed unexpectedly")

# Start the WebSocket server
start_server = websocket.serve(test_ws_handler, "127.0.0.1", 5000)  # Adjust the WebSocket server address and port
print("WebSocket server started")

# Main loop for real-time gesture recognition
while True:
    success, img = cap.read()
    # Check if the frame is successfully read
    if not success:
        print("Failed to read a frame. Please check if your webcam is in use. Exiting...")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    try:
        if hands:
            if len(hands) == 1:
                # Process one hand and make predictions for one-handed gestures
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Draw rectangle for one hand
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

                try:
                    # Crop and resize hand for gesture recognition
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgCrop1 = img[y - offset:y + h + offset, x - offset:x + w + offset]
                    imgResize1 = cv2.resize(imgCrop1, (imgSize, imgSize))
                    imgWhite[:imgSize, :] = imgResize1

                    # Get predictions and update display
                    prediction, index = classifier_one_handed.getPrediction(imgWhite, draw=False)
                    cv2.putText(imgOutput, labels_one_handed[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1.7,
                                (0, 0, 0), 2)

                    # Update statistics for one-handed gestures
                    total_frames_one_handed += 1
                    ground_truth_index = labels_one_handed.index(labels_one_handed[index])
                    if index == ground_truth_index:
                        correct_predictions_one_handed[index] += 1

                    accuracy = (correct_predictions_one_handed[index] / total_frames_one_handed) * 100
                    confidence_one_handed[index] = prediction[index] * 100
                    cumulative_accuracy_one_handed += accuracy

                    # Display accuracy and confidence
                    cv2.putText(imgOutput, f"Accuracy: {accuracy:.2f}%", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 0, 0), 2)
                    cv2.putText(imgOutput, f"Confidence: {confidence_one_handed[index]:.2f}%", (10, 70),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

                except Exception as e:
                    print(f"An error occurred in processing one hand: {e}")

                # Return the processed result to the server
                result_one_handed = {
                    'output': labels_one_handed[index],
                    'accuracy': accuracy,
                    'confidence': confidence_one_handed[index]
                }
                result_json_one_handed = json.dumps(result_one_handed)
                response_one_handed = requests.post(server_url, data=result_json_one_handed, headers=headers)
                print(response_one_handed.text)

            elif len(hands) == 2:
                # Process two hands and make predictions for two-handed gestures
                hand1, hand2 = hands[0], hands[1]
                x1, y1, w1, h1 = hand1['bbox']
                x2, y2, w2, h2 = hand2['bbox']

                # Calculate distance between hands
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if distance < join_distance_threshold:
                    # If hands are close, consider them as one
                    x, y, w, h = min(x1, x2), min(y1, y2), max(w1, w2), max(h1, h2)

                    # Draw rectangle for joined hands
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

                    try:
                        # Crop and resize joined hands for gesture recognition
                        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                        imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
                        imgWhite[:imgSize, :imgSize // 2] = imgResize[:, :imgSize // 2]  # Left hand
                        imgWhite[:imgSize, imgSize // 2:] = imgResize[:, imgSize // 2:]  # Right hand

                        # Get predictions and update display
                        prediction, index = classifier_two_handed.getPrediction(imgWhite, draw=False)
                        cv2.putText(imgOutput, labels_two_handed[index], (int((x1 + x2) / 2), int((y1 + y2) / 2) - 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0), 2)

                        # Update statistics for two-handed gestures
                        total_frames_two_handed += 1
                        ground_truth_index = labels_two_handed.index(labels_two_handed[index])
                        if index == ground_truth_index:
                            correct_predictions_two_handed[index] += 1

                        accuracy = (correct_predictions_two_handed[index] / total_frames_two_handed) * 100
                        confidence_two_handed[index] = prediction[index] * 100
                        cumulative_accuracy_two_handed += accuracy

                        # Display accuracy and confidence
                        cv2.putText(imgOutput, f"Accuracy: {accuracy:.2f}%", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 0, 0), 2)
                        cv2.putText(imgOutput, f"Confidence: {confidence_two_handed[index]:.2f}%", (10, 70),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

                    except Exception as e:
                        print(f"An error occurred in processing joined hands: {e}")

                    # Return the processed result to the server
                    result_two_handed = {
                        'output': labels_two_handed[index],
                        'accuracy': accuracy,
                        'confidence': confidence_two_handed[index]
                    }
                    result_json_two_handed = json.dumps(result_two_handed)
                    response_two_handed = requests.post(server_url, data=result_json_two_handed, headers=headers)
                    print(response_two_handed.text)

                else:
                    # If hands are not close, draw individual rectangles for each hand
                    cv2.rectangle(imgOutput, (x1 - offset, y1 - offset), (x1 + w1 + offset, y1 + h1 + offset),
                                  (255, 0, 255), 4)
                    cv2.rectangle(imgOutput, (x2 - offset, y2 - offset), (x2 + w2 + offset, y2 + h2 + offset),
                                  (255, 0, 255), 4)

                    try:
                        # Crop and resize individual hands for gesture recognition for hand1
                        imgWhite1 = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                        imgCrop1 = img[y1 - offset:y1 + h1 + offset, x1 - offset:x1 + w1 + offset]
                        imgResize1 = cv2.resize(imgCrop1, (imgSize, imgSize))
                        imgWhite1[:imgSize, :] = imgResize1

                        # Get predictions and update display for hand1
                        prediction1, index1 = classifier_one_handed.getPrediction(imgWhite1, draw=False)
                        cv2.putText(imgOutput, labels_one_handed[index1], (x1, y1 - 30), cv2.FONT_HERSHEY_COMPLEX,
                                    1.7, (0, 0, 0), 2)

                        # Update statistics for one-handed gestures for hand1
                        total_frames_one_handed += 1
                        ground_truth_index1 = labels_one_handed.index(labels_one_handed[index1])
                        if index1 == ground_truth_index1:
                            correct_predictions_one_handed[index1] += 1

                        accuracy1 = (correct_predictions_one_handed[index1] / total_frames_one_handed) * 100
                        confidence_one_handed[index1] = prediction1[index1] * 100
                        cumulative_accuracy_one_handed += accuracy1

                        # Display accuracy and confidence for hand1
                        cv2.putText(imgOutput, f"Accuracy: {accuracy1:.2f}%", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 0, 0), 2)
                        cv2.putText(imgOutput, f"Confidence: {confidence_one_handed[index1]:.2f}%", (10, 70),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

                    except Exception as e:
                        print(f"An error occurred in processing hand1: {e}")

                    try:
                        # Crop and resize individual hands for gesture recognition for hand2
                        imgWhite2 = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                        imgCrop2 = img[y2 - offset:y2 + h2 + offset, x2 - offset:x2 + w2 + offset]
                        imgResize2 = cv2.resize(imgCrop2, (imgSize, imgSize))
                        imgWhite2[:imgSize, :] = imgResize2

                        # Get predictions and update display for hand2
                        prediction2, index2 = classifier_one_handed.getPrediction(imgWhite2, draw=False)
                        cv2.putText(imgOutput, labels_one_handed[index2].split()[1], (x2, y2 - 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0), 2)

                        # Update statistics for one-handed gestures for hand2
                        total_frames_one_handed += 1
                        ground_truth_index2 = labels_one_handed.index(labels_one_handed[index2])
                        if index2 == ground_truth_index2:
                            correct_predictions_one_handed[index2] += 1

                        accuracy2 = (correct_predictions_one_handed[index2] / total_frames_one_handed) * 100
                        confidence_one_handed[index2] = prediction2[index2] * 100
                        cumulative_accuracy_one_handed += accuracy2

                        # Display accuracy and confidence for hand2
                        cv2.putText(imgOutput, f"Accuracy: {accuracy2:.2f}%", (10, 130), cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 0, 0), 2)
                        cv2.putText(imgOutput, f"Confidence: {confidence_one_handed[index2]:.2f}%", (10, 170),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

                    except Exception as e:
                        print(f"An error occurred in processing hand2: {e}")

                # Display cumulative accuracy over time for one-handed and two-handed gestures
                avg_accuracy_one_handed = cumulative_accuracy_one_handed / total_frames
                avg_accuracy_two_handed = cumulative_accuracy_two_handed / total_frames

                cv2.putText(imgOutput, f"Avg Accuracy (One-handed): {avg_accuracy_one_handed:.2f}%", (10, 110),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(imgOutput, f"Avg Accuracy (Two-handed): {avg_accuracy_two_handed:.2f}%", (10, 150),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    except Exception as e:
        print(f"An error occurred: {e}")

    # Display the output image
    cv2.imshow("Image", imgOutput)

    # Exit the program when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()