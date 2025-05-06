import face_recognition
import cv2
image = face_recognition.load_image_file("/home/zzy/ComputerVision/Computer-Vision-Project/test.png")
face_locations = face_recognition.face_locations(image)
print(face_locations)
for face_location in face_locations:
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
# to visualize the detected faces   
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow("Detected Faces", rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    # You can also draw rectangles around the faces using OpenCV or any other library