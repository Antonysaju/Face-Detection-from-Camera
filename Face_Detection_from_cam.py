import cv2

cascPath= "haarcascade_frontalface_default.xml"
faceCascade= cv2.CascadeClassifier(cascPath)

video_capture= cv2.VideoCapture(0)

process_this_frame= True

while True:
    ret, frame= video_capture.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    small_frame= cv2. resize(frame, (0,0), fx= 0.25, fy= 0.25)
    rgb_small_frame= cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    process_this_frame= not process_this_frame

    faces= faceCascade.detectMultiScale(
        gray,
        scaleFactor= 1.1,
        minNeighbors= 5,
        minSize= (30,30),
        flags= cv2.CASCADE_SCALE_IMAGE
    )

    nfaces= 0
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        nfaces= nfaces+1

    cv2.imshow("Faces found", frame)
    if cv2.waitKey(1) & 0xFF== ord("q"):
        break

print("Found {0} faces".format(nfaces))
video_capture.release()
cv2.destroyAllWindows()
