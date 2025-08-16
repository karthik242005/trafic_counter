import cv2
from ultralytics import YOLO
import cv_utils
from mot_sort import  *
import matplotlib.pyplot as plt

# cam = cv2.VideoCapture
cam = cv2.VideoCapture("data/Cars Moving On Road Stock Footage - Free Download.mp4")
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



#load model. (just type model name and it will download automaticaly)
model = YOLO(model='yolo-weights/yolov8l.pt')
class_names = model.names #COCO classes


mask_img = cv2.imread("data/mask.png")
#max_age = frame son
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
total_count = []

while True:

    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (1280,720))
    #bitwise orqali faqatgina aniq maskadagi white qismni framemizdan qirqib modelga uzatamiz.
    masked = cv2.bitwise_and(frame, mask_img)
    line_cor = [130,400, 1220,400]
    #predict by yolo model
    predictions = model(masked, stream=True)
    detections = np.empty((0, 5))  # Return a new array of given shape and type, without initializing entries.

    for r in predictions:
        boxes = r.boxes

        #loop throught detected objects
        for box in boxes:
            cls = int(box.cls.item()) #class number for object
            conf = round(box.conf.item(), 2) #confidence threshold. tensor qaytaradi

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1


            if (class_names[cls] == "car" or class_names[cls] == 'motorcycle' or class_names[cls] == 'truck'
                    or class_names[cls] == 'bus' and conf > 0.4):

                #cv_utils.cornerRect(frame, [x1, y1, w, h], t=2, rt=1)
                #cv_utils.putTextRect(frame, text=f"{class_names[cls]} {conf}",pos=(x1, y1), scale=2, thickness=1)

                # current_array - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
                current_array = np.array([x1,y1,int(x2),int(y2), conf])
                # vertical holatda stack qilib listga oxshatib qoshob ketadi
                detections = np.vstack((detections,current_array))

    track_results = tracker.update(detections) #update track id larni yanagi safar korsatmaydi
   # cv2.line(frame, (line_cor[0], line_cor[1]), (line_cor[2], line_cor[3]), color=(255, 0, 0), thickness=10)

    for res in track_results:
        x1, y1, x2, y2, id = res
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h  = x2 - x1, y2 - y1
        cx, cy = x1 + w //2, y1 + h //2
        cv2.circle(frame, center=(cx, cy), radius=3, color=(0,0,255), thickness=3)
        cv_utils.cornerRect(frame, [x1, y1, w, h], t=2, rt=1, colorR=(255,0,0))
        #cv_utils.putTextRect(frame, text=f"{int(id)}", pos=(max(0, x1), max(35,y1)), scale=2, thickness=1, offset=10)

        if line_cor[0]<cx<line_cor[2] and line_cor[1]-10<cy<line_cor[3]+10:
            if id not in total_count: total_count.append(id)

    cv_utils.putTextRect(frame, text=f"Count: {len(total_count)}", pos=(20,50), scale=2, thickness=1, offset=10)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()