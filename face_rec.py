import cv2
import dlib

# specify detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# captures video from camera
video_capture = cv2.VideoCapture(0)
#to capture video from local directory use
#video_capture = cv2.VideoCapture('<local path to video>')

# success = True if video is captured, else false. 
success, ret = video_capture.read()
count = 1
while success:
# Extract frames from video using read()
    success, ret = video_capture.read()
# Resizing improves speed of face detection
    ret = cv2.resize(ret, (0,0), fx=0.5, fy=0.5) 

    if count%10 ==0:
# frame is given as input to detector
      faces = detector(ret,1)
#For each detected face
      for k,d in enumerate(faces): 
#Get coordinates
          shape = predictor(ret, d) 
#There are 68 landmark points on each face
          for i in range(1,68): 
                #cv2.circle(ret, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)
                # Draw rectangle around each face
                cv2.rectangle(ret, (d.left(),d.top()), (d.right(), d.bottom()), (255, 255, 255), 1)

      ret = cv2.resize(ret, (0,0), fx=0.8, fy=0.8) 
#Display the frame
      cv2.imshow("image", ret) 
 
    count=count+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
