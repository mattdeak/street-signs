import cv2
from lib.extractor import process_image
import tensorflow as tf
import os

PREVIEW = False
MODEL_DIR = os.path.join(os.getcwd(), "saved_models")

model_name = "vggpretrained01_noise_extra.hp5"
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, model_name))
# mser = cv2.MSER_create(_delta=20, _min_area=400, _max_area=10000)
mser = cv2.MSER_create(_delta=12, _min_area=100, _max_area=50000)

video_file = "video/printvid1.mp4"
output_file = "video_out.avi"
cap = cv2.VideoCapture(video_file)

ret, frame = cap.read()

h, w, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(output_file, fourcc, 30, (w, h))

if PREVIEW:
    window = cv2.namedWindow("window")

try:
    while cap.isOpened():
        if ret:
            processed_frame = process_image(
                frame, mser, model, write_mser_detected=False, threshold=0.75
            )

            writer.write(processed_frame)

        ret, frame = cap.read()
        if not ret:
            break

        if PREVIEW:
            cv2.imshow("window", processed_frame)

            while PREVIEW and ret:
                if cv2.waitKey(25) == ord("q"):
                    break
finally:
    cv2.destroyAllWindows()
    cap.release()
    writer.release()
