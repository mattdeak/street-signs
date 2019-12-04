from lib.extractor import *
import ntpath
import cv2
import tensorflow as tf
import os


ANNOTATION_OFFSET = 50

INPUT_DIR = os.path.join(os.getcwd(), "input")
MODEL_DIR = os.path.join(os.getcwd(), "saved_models")
OUTPUT_DIR = os.path.join(os.getcwd(), "graded_images")


def draw_number_box(image, bbox, cls, conf):
    x, y, w, h = bbox
    text_location = (x, y)

    # This is the bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # This is a background for the class label
    cv2.rectangle(
        image, (x, y - ANNOTATION_OFFSET), (x + w, y), (0, 255, 0), thickness=cv2.FILLED
    )

    # Put text in background
    text = f"{cls}"

    cv2.putText(
        image, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2,
    )


def draw_final_house_number(image, box_records):
    """Draws the boxes as they appear left-to-right or top-to-bottom"""
    orientation = determine_orientation(box_records)

    if orientation == "vertical":
        # If vertical, sort by y value
        sorted_records = sorted(box_records, key=lambda x: x[2][1])
    else:
        # If horizontal, sort by x value
        sorted_records = sorted(box_records, key=lambda x: x[2][0])

    # Extracts the class predictions in order of their appearance and
    # concatenates them into a final string
    final_number = "".join([str(record[1]) for record in sorted_records])

    # Print final number
    text = f"House Number: {final_number}"

    h, w, _ = image.shape

    cv2.putText(
        image,
        text,
        (0, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        5,
        color=(0, 0, 0),
        thickness=15,
    )
    cv2.putText(
        image,
        text,
        (0, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        5,
        color=(255, 255, 255),
        thickness=7,
    )


def determine_orientation(box_records):
    """Determines the orientation of the house number (primarily horizontal or primarily vertical)"""
    # TODO: Sort by x value, take 1st - last. Compare to sorted by y value
    # smaller value is the dominant orientation. (E.g smaller x1 - x4 == vertical orientation)
    bboxes = [record[2] for record in box_records]
    x_sorted = sorted(bboxes, key=lambda x: x[0])
    y_sorted = sorted(bboxes, key=lambda x: x[1])

    x_diff = abs(x_sorted[0][0] - x_sorted[-1][0])
    y_diff = abs(y_sorted[0][1] - y_sorted[-1][1])

    if x_diff > y_diff:
        return "horizontal"
    else:
        return "vertical"


if __name__ == "__main__":

    model_name = "vggpretrained01_noise_extra.hp5"
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, model_name))
    mser = cv2.MSER_create(_delta=12, _min_area=400, _max_area=25000)

    filepaths = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)]
    for f in filepaths:
        filename = ntpath.basename(f)
        filename = filename.replace(".jpeg", "").replace(".png", "").replace(".jpg", "")
        write_path = os.path.join(OUTPUT_DIR, f"{filename}.png")

        print(f"Processing file: {filename}")

        img = cv2.imread(f)

        processed_img = process_image(img, mser, model)
        cv2.imwrite(write_path, processed_img)
