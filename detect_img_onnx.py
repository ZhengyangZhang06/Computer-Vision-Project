"""
This code uses the onnx model to detect faces from live video or cameras.
"""
import os
import time

import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils

# onnx runtime
import onnxruntime as ort


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                      iou_threshold=iou_threshold,
                                      top_k=top_k,
                                      )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


label_path = "models/voc-model-labels.txt"
onnx_path = "models/onnx/version-RFB-320.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]

# Load ONNX model - remove caffe2 dependency
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name
result_path = "./detect_imgs_results_onnx"

threshold = 0.7
path = "/home/zzy/ComputerVision/Computer-Vision-Project/test.png"
# Add this code to the end of the file to process and display results

# Load and process the image
orig_image = cv2.imread(path)
if orig_image is None:
    print(f"Error: Could not load image from {path}")
    exit(-1)
    
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (320, 240))
image_mean = np.array([127, 127, 127])
image = (image - image_mean) / 128
image = np.transpose(image, [2, 0, 1])
image = np.expand_dims(image, axis=0)
image = image.astype(np.float32)

# Run inference with ONNX Runtime
confidences, boxes = ort_session.run(None, {input_name: image})

# Process the raw output
width, height = orig_image.shape[1], orig_image.shape[0]
boxes, labels, probs = predict(width, height, confidences, boxes, threshold)

# Visualize the results
result_image = orig_image.copy()
for i in range(boxes.shape[0]):
    box = boxes[i, :]
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    
    # Draw bounding box
    cv2.rectangle(result_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
    # Draw label
    cv2.putText(result_image, label,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # font scale
                (0, 255, 0),  # green color
                2)  # thickness

print(f"Found {len(boxes)} faces")

# Create result directory if it doesn't exist
os.makedirs(result_path, exist_ok=True)

# Save the result
output_path = os.path.join(result_path, os.path.basename(path))
cv2.imwrite(output_path, result_image)
print(f"Result saved to {output_path}")

# Display the result
cv2.imshow("Face Detection Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()