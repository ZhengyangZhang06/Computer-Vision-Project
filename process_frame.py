from facial_emotion_analysis import process_frame_with_tracking
import cv2
import dlib
import torch
from models.resnet import ResNet18
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
model_path = 'models/fer2013_resnet_best.pth'
frame = cv2.imread("test.jpg")
detector = dlib.get_frontal_face_detector()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet18().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
processed_frame, results = process_frame_with_tracking(
                    frame.copy(), detector, predictor, model, DEVICE, previous_face_regions
                )
previous_results = results

# Update previous_face_regions for the next frame
previous_face_regions = [(r['bbox'][0], r['bbox'][1], 
r['bbox'][0] + r['bbox'][2], 
r['bbox'][1] + r['bbox'][3]) 
for r in results]
cv2.imshow("Processed Frame", processed_frame)
cv2.waitKey(1)
