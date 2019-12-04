import cv2
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import tensorflow as tf

ASPECT_RATIO_MIN = 0.3
ASPECT_RATIO_MAX = 1.2

ANNOTATION_OFFSET = 50


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
            overlap1 = overlap_ratio(box1, box2)
            overlap2 = overlap_ratio(box2, box1)
            if overlap1 > threshold or overlap2 > threshold:
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


def get_detections(bboxes, img, model, threshold=0.9, write_all_boxes=False):
    detections = []
    cropped_images = []
    boxes = []
    for box in tqdm(bboxes):
        x, y, w, h = box
        if write_all_boxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cropped = img[y : y + h, x : x + w]

        # Use cubic interpolation to grow, area to shrink
        if w > 32 or h > 32:
            resized = cv2.resize(cropped, (32, 32), cv2.INTER_CUBIC).astype(float)
        else:
            resized = cv2.resize(cropped, (32, 32), cv2.INTER_AREA).astype(float)
        cropped_images.append(resized)

    cropped_images = np.array(cropped_images)
    preds = model.predict(cropped_images / 255.0)

    for pred, box in zip(preds, bboxes):
        if any(pred[:10] > threshold):
            predicted_class = pred.argmax() + 1  # Because of weird digit handling
            if predicted_class == 10:
                predicted_class = 0
            box_record = (pred.max(), predicted_class, box)
            detections.append(box_record)

    return detections


def get_centroid(bbox):
    x, y, w, h = bbox
    return (x + w // 2), (y + h // 2)


def get_distance(bbox1, bbox2):
    x1, y1 = get_centroid(bbox1)
    x2, y2 = get_centroid(bbox2)

    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def prune_by_distance(box_records, multiplier=1.5):
    pruned_boxes = set()
    for i in range(len(box_records) - 1):
        _, _, box1 = box_records[i]
        for j in range(i + 1, len(box_records)):
            _, _, box2 = box_records[j]
            distance = get_distance(box1, box2)
            size1 = max(box1[2], box1[3])
            size2 = max(box2[2], box2[3])
            if distance < min(size1, size2) * multiplier:
                pruned_boxes.add(i)
                pruned_boxes.add(j)

    return [box_records[i] for i in list(pruned_boxes)]


def process_image(
    img, mser, model, write_mser_detected=False, prune_aspect_ratio=True, threshold=0.85
):
    img_copy = img.copy()
    rgb_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray_image

    _, bboxes1 = mser.detectRegions(gray_image)
    _, bboxes2 = mser.detectRegions(inverted_gray)

    bboxes = np.vstack([bboxes1, bboxes2])

    if write_mser_detected:
        for box in bboxes:
            x, y, w, h = box
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if prune_aspect_ratio:
        for i in range(len(bboxes) - 1, -1, -1):
            bbox = bboxes[i]
            x, y, w, h = bbox
            aspect_ratio = w / h
            if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
                bboxes = np.delete(bboxes, i, 0)

    if write_mser_detected:
        for box in bboxes:
            x, y, w, h = box
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

    try:
        detections = get_detections(bboxes, rgb_image, model, threshold=threshold)
    except Exception as e:
        print(e)
        return img_copy

    suppressed_detections = nms(detections)
    pruned_detections = prune_by_distance(suppressed_detections)
    print(f"Writing {len(pruned_detections)} boxes")

    if not pruned_detections:
        print("No digits detected. Skipping...")
        return img_copy

    for confidence, cls, bbox in pruned_detections:
        draw_number_box(img_copy, bbox, cls, confidence)

    draw_final_house_number(img_copy, pruned_detections)
    return img_copy


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
