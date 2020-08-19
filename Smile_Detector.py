import cv2

#Face and Smile Classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#viela_jones
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#Grab the Webcam Feed
webcam = cv2.VideoCapture(0) #To capture the Video Feed

# show the current frame
while True:

    successful_frame_read, frame = webcam.read() #reading from the webcam feed
    
        
    #If there's an error, abort
    if not successful_frame_read:
        break

    #Change to grayscale
    #Using a gray_Scale to optimize the video processing as the color feed has 3 channels, but black and white feed has 1 channel
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#convert the color of the image
    
    #Detect Faces First
    faces = face_detector.detectMultiScale(frame_grayscale)#returns the array of points of face_location
    


    #Run Face Detection from the live feed
    
    for (x,y,w,h) in faces:

        #Draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (100,200,50),4)

        # Get the sub frame (Using numpy as N-dimensional array slicing)
        face = frame[y:y+h, x:x+w] # This will not work with vanilla python as opencv is based on numpy
    

        #change the gray_scale
        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayscale,1.7,20)

        """
        # Find all smiles in the face
        for(x1,y1,w1,h1) in smiles:
            # Draw all the rectangles around the smile
            cv2.rectangle(face, (x1,y1), (x1 + w1,y1 + h1), (50,50,200),4)
        """

        # Label this face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x,y+h+40),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))


    """
    #Run Smile Detection within each of those faces
    for (x,y,w,h) in smiles:

        #Draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,200),4)
    """

    cv2.imshow('Smile Detector', frame) #to get the current frame (Single Frame)
    

    #Display (Wait until you hide it)
    cv2.waitKey(1)

#CleanUp
webcam.release()
cv2.destroyAllWindows()