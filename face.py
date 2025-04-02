import cv2
import dlib
import argparse
import numpy as np

def detect_faces_with_landmarks(image_path, save_output=False):
    # Load face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # Download this file from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    print(f"Found {len(faces)} faces in the image.")
    
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)
        
        # Create a complete face outline using all relevant landmark points
        
        # 1. Get the jawline points (0-16)
        jawline_points = []
        for i in range(0, 17):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            jawline_points.append((x, y))
        
        # 2. Get the eyebrow points (17-26)
        eyebrow_points = []
        for i in range(17, 27):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            eyebrow_points.append((x, y))
            
        # 3. Get the nose bridge points (27-30)
        nose_bridge = []
        for i in range(27, 31):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            nose_bridge.append((x, y))
            
        # 4. Get eyes points (36-47)
        left_eye = []
        for i in range(36, 42):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            left_eye.append((x, y))
            
        right_eye = []
        for i in range(42, 48):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            right_eye.append((x, y))
        
        # Draw all the facial features
        # 1. Draw jawline
        jawline_points = np.array(jawline_points, np.int32)
        cv2.polylines(image, [jawline_points], False, (0, 255, 0), 2)
        
        # 2. Draw eyebrows
        eyebrow_points = np.array(eyebrow_points, np.int32)
        for i in range(0, 5):
            cv2.line(image, 
                    (eyebrow_points[i][0], eyebrow_points[i][1]),
                    (eyebrow_points[i+1][0], eyebrow_points[i+1][1]),
                    (0, 255, 0), 2)
        for i in range(5, 9):
            cv2.line(image, 
                    (eyebrow_points[i][0], eyebrow_points[i][1]),
                    (eyebrow_points[i+1][0], eyebrow_points[i+1][1]),
                    (0, 255, 0), 2)
        
        # 3. Draw nose bridge
        nose_bridge = np.array(nose_bridge, np.int32)
        cv2.polylines(image, [nose_bridge], False, (0, 255, 0), 2)
        
        # 4. Draw eyes
        left_eye = np.array(left_eye, np.int32)
        right_eye = np.array(right_eye, np.int32)
        cv2.polylines(image, [left_eye], True, (0, 255, 0), 2)
        cv2.polylines(image, [right_eye], True, (0, 255, 0), 2)
        
        # 5. Alternatively, draw a rectangle around the entire face
        x, y = face.left(), face.top()
        w, h = face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    # Display the result
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the output image if requested
    if save_output:
        output_path = image_path.rsplit('.', 1)[0] + '_detected.jpg'
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect faces in an image.')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--save', action='store_true', help='Save the output image')
    args = parser.parse_args()
    
    detect_faces_with_landmarks(args.image_path, args.save)