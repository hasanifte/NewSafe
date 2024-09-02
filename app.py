import easyocr
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import tempfile
import os

# Initialize EasyOCR reader globally
reader = easyocr.Reader(['en'])

def process_frame(frame):
    try:
        # Convert frame to BGR for OpenCV processing
        image_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # Perform OCR on the entire frame
        results = reader.readtext(image_bgr)

        # List to hold the detected texts
        detected_texts = []

        # Draw bounding boxes and text on the frame
        for (bbox, text, prob) in results:
            detected_texts.append(text)
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
            cv2.rectangle(image_bgr, (int(x1), int(y1)), (int(x3), int(y3)), (0, 255, 0), 2)
            cv2.putText(image_bgr, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert BGR frame back to RGB
        frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Convert the frame back to PIL Image format
        result_image = Image.fromarray(frame_rgb)

        # Join all detected texts into a single string
        result_text = "\n".join(detected_texts)

        return result_image, result_text
    except Exception as e:
        return None, f"Error processing frame: {str(e)}"

def process_video(video_file_path):
    try:
        # Create a temporary file for output
        output_path = os.path.join(tempfile.gettempdir(), 'output.avi')

        # Open the video file
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file.")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process each frame
        detected_texts = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result_image, result_text = process_frame(frame)
            if result_image is None:
                raise ValueError(result_text)  # Pass error message from frame processing
            detected_texts.append(result_text)
            out.write(np.array(result_image))

        cap.release()
        out.release()

        # Join all detected texts into a single string
        full_text = "\n".join(detected_texts)

        return output_path, full_text
    except Exception as e:
        return None, f"Error processing video: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(),  # Input video file
    outputs=[gr.Video(), gr.Textbox()]  # Output video file and detected text
)

# Launch the interface
iface.launch()
