import cv2
import matplotlib.pyplot as plt

# Set up Model
#mobilenet is a tensorflow mobile model, mobilenet is a class of CNN opensource by google. Good starting point. 
#Coco dataset is a good dataset for starting (common object and context) - 1.5 million objects
## Coco dataset is a good starting point
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)
classLabels = []
file_name = "labels.txt"
with open(file_name, 'rt') as fpt:
    classLabels= fpt.read().rstrip('\n').split('\n')

model.setInputSize(320,320) #Size of new frame
model.setInputScale(1.0/127.5) #Scale factor of the value for the frame, multiplier for frame values
model.setInputMean((127.5,127.5,127,5))
model.setInputSwapRB(True)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

# Read photo from file
def read_photo(filename):
    img = cv2.imread(filename)
    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(img, boxes, (255,0,0), 2)
        cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color=(0,255,0), thickness=10)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

# Take photo from WebCam
def take_photo(): 
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite('webcamphoto.jpg', frame)
    cap.release()

# Stream video
def stream_video(): 
    cap = cv2.VideoCapture(0) # Connect to webcam

    # Establish loop through every frame until webcam close
    while cap.isOpened():
        # Read image
        ret, frame = cap.read()

        # Show image
        cv2.imshow("Webcam", frame)

        # Checks whether q has been hit and stops the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release() # release webcam
    cv2.destroyAllWindows() # closes frame

def stream_video_with_detect():
    cap = cv2.VideoCapture(0) # Connect to webcam

    # Establish loop through every frame until webcam close
    while cap.isOpened():
        # Read image
        ret, frame = cap.read()

        # Process frame through detection model
        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.6)
        if (len(ClassIndex)!=0):
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                #print(ClassInd)
                cv2.rectangle(frame, boxes, (255,0,0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color=(0,255,0), thickness=1)

            # Show image
        cv2.imshow("Webcam", frame)

        # Checks whether q has been hit and stops the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release() # release webcam
    cv2.destroyAllWindows() # closes frame

stream_video_with_detect()

