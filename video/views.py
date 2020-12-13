from os.path import normpath, join

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage
import os


# Create your views here.

def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)


def finalVid(name):
    # import the necessary packages
    import numpy as np
    import imutils
    import time
    import cv2
    import json

    print("program started")

    labelsPath = "./models/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    weightsPath = "./models/yolov3.weights"
    configPath = "./models/yolov3.cfg"

    print("all file loaded")

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print(os.getcwd())
    filePath = join(os.getcwd(), "media", name)
    print(filePath)

    vs = cv2.VideoCapture(filePath)

    type(vs)
    print(vs)
    writer = None
    (W, H) = (None, None)

    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    except:
        print("[INFO] could not determine of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
        freq = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # freq = [boxes.count(i) for i in boxes]

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

        # if len(idxs) > 0:
        # 	# loop over the indexes we are keeping
        # 	var1 = 0
        # 	var = []
        # 	for i in idxs.flatten():
        # 		# extract the bounding box coordinates
        # 		(x, y) = (boxes[i][0], boxes[i][1])
        # 		(w, h) = (boxes[i][2], boxes[i][3])
        #
        # 		# draw a bounding box rectangle and label on the frame
        # 		color = [int(c) for c in COLORS[classIDs[i]]]
        # 		cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # 		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        # 		cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # 		cv2.line(frame, (415, 361), (615, 361), (0, 0, 0xFF), 3)
        # 		cv2.line(frame, (670, 361), (843, 361), (0, 0, 0xFF), 3)
        # 		freq1 = [j for j in classIDs]
        # 		#freq = [[LABELS[classIDs[x]], classIDs.count(x)] for x in set(classIDs)]
        # 		freq = dict([LABELS[x], classIDs.count(x)] for x in set(classIDs))
        # 		#print("Class:", freq, freq1)
        # 		freq = str(freq)[1:-1]
        # 		text1 = ("Overall Vehicles in Frame = {}".format(freq))
        # 		cv2.putText(frame, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 2)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.line(frame, (415, 361), (615, 361), (0, 0, 0xFF), 3)
                cv2.line(frame, (670, 361), (843, 361), (0, 0, 0xFF), 3)
                freq1 = [j for j in classIDs]
                freq = dict([LABELS[x], classIDs.count(x)] for x in set(classIDs))
                freq = str(freq)[1:-1]
                text1 = ("Overall Vehicles in Frame = {}".format(freq))
                cv2.putText(frame, text1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 2)

                # freq = len([boxes.count(i) for i in boxes])
                # # freq = len(boxes[i])
                # text1 = "Number of objects in the frame: {}".format(freq)
                # cv2.putText(frame, text1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            # self._name = name + '.mp4'
            # self._cap = VideoCapture(0)
            # self._fourcc = VideoWriter_fourcc(*'MP4V')
            # self._out = VideoWriter(self._name, self._fourcc, 20.0, (640, 480))
            # fourcc = cv2.VideoWriter_fourcc(*'X264')
            writer = cv2.VideoWriter("./media/final.mp4", 0x00000021, 20, (frame.shape[1], frame.shape[0]), True)

            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))

        # write the output frame to disk
        writer.write(frame)

    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()


def predict(request):
    fileObj = request.FILES['file1']
    name = default_storage.save(fileObj.name, fileObj)
    filePathName = default_storage.url(name)
    name1 = default_storage.open(name)
    print(name)
    print(filePathName)
    print(name1)

    finalVid(name)

    context = {'name': filePathName}
    return render(request, 'index.html', context)
