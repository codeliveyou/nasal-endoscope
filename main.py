import cv2
import numpy as np
import argparse
import sys
import time
import threading
import tensorflow as tf

class InferenceThread(threading.Thread):
    def __init__(self, model_path, input_details, output_details):
        threading.Thread.__init__(self)
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = input_details
        self.output_details = output_details
        self.frame = None
        self.prediction = None
        self.lock = threading.Lock()
        self.running = True

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame

    def get_prediction(self):
        with self.lock:
            return self.prediction

    def run(self):
        while self.running:
            if self.frame is not None:
                # Copy the frame to avoid modification during processing
                with self.lock:
                    frame_to_process = self.frame.copy()
                    self.frame = None  # Reset frame

                # Preprocess frame
                input_data = preprocess_frame(frame_to_process)

                # Perform inference
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])

                # Update prediction
                with self.lock:
                    self.prediction = prediction

            else:
                # Sleep briefly to avoid busy waiting
                time.sleep(0.01)

    def stop(self):
        self.running = False

def preprocess_frame(frame, img_height=224, img_width=224):
    # Resize the frame
    frame_resized = cv2.resize(frame, (img_width, img_height))
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to [0, 1]
    frame_normalized = frame_rgb.astype('float32') / 255.0
    # Expand dimensions to match model's input shape
    input_data = np.expand_dims(frame_normalized, axis=0)
    return input_data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Nasal Endoscope Real-Time Classification')
    parser.add_argument('--mode', type=str, required=True, choices=['test', 'camera'],
                        help='Mode to run the script: test or camera')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the TensorFlow Lite model (.tflite file)')
    parser.add_argument('--video', type=str,
                        help='Path to the input video file for test mode')
    parser.add_argument('--output', type=str, default='output_in_frames.avi',
                        help='Path to the output video file')
    args = parser.parse_args()

    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    print(f"Loaded TFLite model from {args.model}")

    # Get model input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_height = input_details[0]['shape'][1]
    img_width = input_details[0]['shape'][2]

    # Initialize video source depending on the mode
    if args.mode == 'test':
        if not args.video:
            print("Error: --video argument is required in test mode.")
            sys.exit(1)
        video_source = args.video
    elif args.mode == 'camera':
        video_source = 0  # Default camera index

    # Open video capture
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        sys.exit(1)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640  # Default width if unable to get
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480  # Default height if unable to get
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video == 0 or np.isnan(fps_video):  # For camera input, fps may not be available
        fps_video = 30  # Default FPS

    # Define VideoWriter object to save 'in' frames
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps_video, (frame_width, frame_height))
    print(f"Output video will be saved to {args.output}")

    # Create and start the inference thread
    inference_thread = InferenceThread(args.model, input_details, output_details)
    inference_thread.start()

    # Variables to keep track of the latest prediction
    latest_prediction = None
    label = 'Processing...'
    color = (255, 255, 255)  # White color for initial state

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.mode == 'test':
                    print("Reached end of video.")
                else:
                    print("Failed to grab frame.")
                break

            # Send the current frame to the inference thread if it's ready for a new frame
            if inference_thread.frame is None:
                inference_thread.set_frame(frame)

            # Get the latest prediction from the inference thread
            prediction = inference_thread.get_prediction()
            if prediction is not None:
                if prediction[0][0] > 0.5:
                    label = 'Out'
                    color = (0, 0, 255)  # Red color in BGR
                    save_frame = False
                else:
                    label = 'In'
                    color = (0, 255, 0)  # Green color in BGR
                    save_frame = True

                # Save frame if classified as 'In'
                if save_frame:
                    out.write(frame)
            else:
                # If no prediction yet, use previous label
                save_frame = False

            # Overlay the label on the frame
            cv2.putText(frame, f"Prediction: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Display the frame
            cv2.imshow('Nasal Endoscope Real-Time Classification', frame)

            # Wait for 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Stop the inference thread
        inference_thread.stop()
        inference_thread.join()

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Resources released. Output video saved.")

if __name__ == '__main__':
    main()

# python main.py --mode test --model trained-models/nasal_endoscope_classifier.h5 --video raw-data/video1.mov --output output_in_frames.avi
# python main.py --mode camera --model trained-models/nasal_endoscope_classifier.h5 --output output_in_frames.avi


