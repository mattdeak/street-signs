import cv2
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import tensorflow as tf

ROOT_DIR = "/home/matt/Projects/CV/final/"
run = True
write_all_boxes = False
show_final_product = False
run_model = False

img_name = 'test1'

filename = [x for x in os.listdir("input") if img_name in x][0] # Filetype agnosticism
print(filename)
test_img = cv2.imread(ROOT_DIR + '/input/' + filename)

model = tf.keras.models.load_model(ROOT_DIR + "saved_models/vggpretrained01_extra.hp5")
# test_img = cv2.resize(test_img, (300, 300))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

mser = cv2.MSER_create(_delta=12, _min_area=40)
_, bboxes = mser.detectRegions(gray_img)


def nms(detections, threshold=0.5):
    """nms

    Parameters
    ----------

    class_boxes : (confidence, (x, y, w, h))
    threshold : threshold for nms

    Returns
    -------
    list of boxes after suppression   
    """

    sorted_boxes = sorted(detections, reverse=True, key=lambda x: x[0])

    i = 0
    j = 1
    marked_for_removal = set()
    while i < len(sorted_boxes) - 1:
        conf, cls, box1 = sorted_boxes[i]
        j = i + 1
        while j < len(sorted_boxes):
            if j in marked_for_removal:
                j += 1
                continue
            _, _, box2 = sorted_boxes[j]
            overlap = overlap_ratio(box1, box2)
            if overlap > threshold:
                marked_for_removal.add(j)
            j += 1
        i += 1

    for ix in sorted(marked_for_removal, reverse=True):
        sorted_boxes.pop(ix)
    return sorted_boxes


def overlap_ratio(bbox1, bbox2):
    """overlap_ratio

    Determines how much of box 2 is inside of box 1
    Parameters
    ----------

    bbox1 :
    bbox2 :

    Returns
    -------
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    left_x_intersect = max(x1, x2)
    top_y_intersect = max(y1, y2)
    right_x_intersect = min(x1 + w1, x2 + w2)
    bottom_y_intersect = min(y1 + h1, y2 + h2)


    intersection = max(0, (right_x_intersect - left_x_intersect)) * max(
        0, bottom_y_intersect - top_y_intersect
    )

    box2_area = ((x2 + w2) - x2) * ((y2 + h2) - y2)
    return intersection / box2_area


def intersect_over_union(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    left_x_intersect = max(x1, x2)
    top_y_intersect = max(y1, y2)
    right_x_intersect = min(x1 + w1, x2 + w2)
    bottom_y_intersect = min(y1 + h1, y2 + h2)

    intersection = max(0, (right_x_intersect - left_x_intersect + 1)) * max(
        0, bottom_y_intersect - top_y_intersect + 1
    )

    box1_area = ((x1 + w1) - x1 + 1) * ((y1 + h1) - y1 + 1)
    box2_area = ((x2 + w2) - x2 + 1) * ((y2 + h2) - y2 + 1)
    union = box1_area + box2_area - intersection
    return intersection / union


def get_detections(bboxes, write_all_boxes=False):
    detections = []
    if run:
        for box in tqdm(bboxes):
            x, y, w, h = box
            if write_all_boxes:
                cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cropped = test_img[y : y + h, x : x + w]
            resized = cv2.resize(cropped, (32, 32)).astype(float)
            blurred = cv2.GaussianBlur(resized, (3, 3), 10)
            pred = model.predict(resized.reshape(1, 32, 32, 3))[0]

            if any(pred[:10] > 0.9):
                predicted_class = pred.argmax() + 1  # Because of weird digit handling
                if predicted_class == 10:
                    predicted_class = 0
                box_record = (pred.max(), predicted_class, box)
                detections.append(box_record)

    return detections


detections = get_detections(bboxes, write_all_boxes=write_all_boxes)
suppressed_detections = nms(detections)

for confidence, cls, bbox in suppressed_detections:
    x, y, w, h = bbox
    text_location = (x, y)
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(
        test_img, str(cls), text_location, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2,
    )

if write_all_boxes:
    cv2.imwrite(f"output/allboxes_{img_name}.jpeg", test_img)
else:
    cv2.imwrite(f"output/result_{img_name}.jpeg", test_img)

if show_final_product:
    window = cv2.namedWindow("window")

    while True:
        cv2.imshow(window, test_img)
        if cv2.waitKey(0) == 27:
            break

cv2.destroyAllWindows()
