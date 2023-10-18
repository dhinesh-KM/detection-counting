from tkinter import *
import tkinter as tk
import tkinter.messagebox as mbox
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np
import os
import imutils
import argparse

NMS_THRESHOLD=0.3
MIN_CONFIDENCE=0.2

# Main Window & Configuration
window = tk.Tk()
window.title("Real Time Human Detection & Counting")
window.iconbitmap('Images/icon.ico')
window.geometry('1000x700')

# top label
start1 = tk.Label(text = "REAL-TIME-HUMAN\nDETECTION  &  COUNTING", font=("Times New Roman", 50,"underline"), fg="black") # same way bg
start1.place(x = 50, y = 10)

# function defined to start the main application
def start_fun():
    window.destroy()

# created a start button
Button(window, text="▶ START",command=start_fun,font=("Arial", 25), bg = "black", fg = "white", cursor="hand2", borderwidth=3, relief="raised").place(x =110 , y =570 )

# image on the main window
path1 = "Images/front1.png"
img2 = ImageTk.PhotoImage(Image.open(path1))
panel1 = tk.Label(window, image = img2)
panel1.place(x = 100, y = 350)

# image on the main window
path = "Images/front2.png"
img1 = ImageTk.PhotoImage(Image.open(path))
panel = tk.Label(window, image = img1)
panel.place(x = 800, y = 350)
exit1 = False
# function created for exiting from window
def exit_win():
    global exit1
    if mbox.askokcancel("Exit", "Do you want to exit?"):
        exit1 = True
        window.destroy()

# exit button created
Button(window, text="❌ EXIT",command=exit_win,font=("Arial", 25), bg = "black", fg = "white", cursor="hand2", borderwidth=3, relief="raised").place(x =790 , y = 570 )
window.protocol("WM_DELETE_WINDOW", exit_win)
window.mainloop()

if exit1==False:
    # Main Window & Configuration of window1
    window1 = tk.Tk()
    window1.title("Real Time Human Detection & Counting")
    window1.iconbitmap('Images/icon.ico')
    window1.geometry('1000x700')

    filename=""
    filename1=""
    filename2=""

    def argsParser():
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
        arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
        arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
        arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
        args = vars(arg_parse.parse_args())
        return args


    # ---------------------------- image section ------------------------------------------------------------
    def image_option():
        # new windowi created for image section
        windowi = tk.Tk()
        windowi.title("Human Detection from Image")
        windowi.iconbitmap('Images/icon.ico')
        windowi.geometry('1000x700')

        # function defined to open the image
        def open_img():
            global filename1
            filename1 = filedialog.askopenfilename(title="Select Image file", parent = windowi)
            path_text1.delete("1.0", "end")
            path_text1.insert(END, filename1)

        def pre_img():
            global filename1
            img = cv2.imread(filename1, 1)
            cv2.imshow("Selected Image Preview", img)

        # function defined to detect the image
        def det_img():
            global filename1
            image_path = filename1
            if(image_path==""):
                mbox.showerror("Error", "No Image File Selected!", parent = windowi)
                return
            info2.config(text="Status : Detecting...")
            mbox.showinfo("Status", "Detecting, Please Wait...", parent = windowi)
            detectByPathImage(image_path)
        
        def detectByPathImage(path):
            global filename1,acc
            image = cv2.imread(path)
            image = imutils.resize(image, width = min(800, image.shape[1]))
            result_image=detect(image)

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        def detect(frame):
            bounding_box_cordinates, weights =  hog.detectMultiScale(frame, winStride = (5, 8), padding = (8, 8), scale = 1.03)
            person = 0
            
            for x,y,w,h in bounding_box_cordinates:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                person += 1
                a=0
                cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
            cv2.putText(frame, f'Total Persons : {person}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
            cv2.imshow('Human Detection from Image', frame)
            a+=person
            mbox.showinfo("TOTAL PERSONS:",a,parent=windowi)
            if a>25:
                mbox.showinfo("Status",'Max. Human Detected is greater than MAX LIMIT.\n Region is Crowded.',parent=windowi)
            else:
                mbox.showinfo("Status",'Max. Human Detected is in the range of MAX LIMIT.\n Region is not Crowded.',parent=windowi)
            info2.config(text="                                                  ")
            info2.config(text="Status : Detection & Counting Completed")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
                # for images ----------------------
        lbl1 = tk.Label(windowi,text="DETECT  FROM IMAGE", font=("Footlight MT Light", 50, "underline"),fg="black")
        lbl1.place(x=180, y=20)
        lbl2 = tk.Label(windowi,text="Selected Image", font=("Verdana", 30),fg="black")
        lbl2.place(x=80, y=200)
        path_text1 = tk.Text(windowi, height=1, width=37, font=("Arial", 30), bg="light grey", fg="black",borderwidth=2, relief="solid")
        path_text1.place(x=80, y = 260)

        Button(windowi, text="SELECT", command=open_img, cursor="hand2", font=("Arial", 20), bg="light blue", fg="black").place(x = 200, y = 350)
        Button(windowi, text="PREVIEW",command=pre_img, cursor="hand2", font=("Arial", 20), bg = "light blue", fg = "black").place(x = 400, y = 350)
        Button(windowi, text="DETECT",command=det_img, cursor="hand2", font=("Arial", 20), bg = "light blue", fg = "black").place(x = 620, y = 350)
        info2 = tk.Label(windowi,font=("Arial", 30), fg="gray")
        info2.place(x=100, y=500)

        def exit_wini():
            if mbox.askokcancel("Exit", "Do you want to exit?", parent = windowi):
                windowi.destroy()
        windowi.protocol("WM_DELETE_WINDOW", exit_wini)
        

# ---------------------------- video section ------------------------------------------------------------
    def video_option():
        # new windowv created for video section
        windowv = tk.Tk()
        windowv.title("Human Detection from Video")
        windowv.iconbitmap('Images/icon.ico')
        windowv.geometry('1000x700')

        # function defined to open the video
        def open_vid():
            global filename2
            filename2 = filedialog.askopenfilename(title="Select Video file", parent=windowv)
            path_text2.delete("1.0", "end")
            path_text2.insert(END, filename2)

          # funcion defined to preview the selected video
        def prev_vid():
            global filename2
            video = cv2.VideoCapture(filename2)
            while (video.isOpened()):
                ret, frame = video.read()
                if ret == True:
                    img = cv2.resize(frame, (800, 500))
                    cv2.imshow('Selected Video Preview', img)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break
            video.release()
            cv2.destroyAllWindows()

            # function defined to detect inside the video
        def det_vid():
            global filename2
            video_path = filename2
            if (video_path == ""):
                mbox.showerror("Error", "No Video File Selected!", parent = windowv)
                return
            info1.config(text="Status : Detecting...")
            mbox.showinfo("Status", "Detecting, Please Wait...", parent=windowv)
    
            args = argsParser()
            writer = None
            if args['output'] is not None:
                writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))
            detectByPathVideo(video_path, writer)

            # the main process of detection in video takes place here
        def detectByPathVideo(path,writer):
            video = cv2.VideoCapture(path)
            check, frame = video.read()
            if check == False:
                mbox.showinfo("Status", "Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).", parent=windowv)
                return
            mbox.showinfo('status','Detecting people...',parent=windowv)
            
            def pedestrian_detection(image, model, layer_name,personidz=0):
                (H, W) = image.shape[:2]
                results = []
                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                             swapRB=True, crop=False)
                model.setInput(blob)
                layerOutputs = model.forward(layer_name)

                boxes = []
                centroids = []
                confidences = []

                for output in layerOutputs:
                    for detection in output:

                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        if classID == personidz and confidence > MIN_CONFIDENCE:

                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            boxes.append([x, y, int(width), int(height)])
                            centroids.append((centerX, centerY))
                            confidences.append(float(confidence))
                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
                # ensure at least one detection exists
                if len(idzs) > 0:
                    # loop over the indexes we are keeping
                    for i in idzs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        # update our results list to consist of the person
                        # prediction probability, bounding box coordinates,
                        # and the centroid
                        res = (confidences[i], (x, y, x + w, y + h), centroids[i])
                        results.append(res)

                #return the list of results
                return results

            labelsPath = "coco.names"
            LABELS = open(labelsPath).read().strip().split("\n")

            weights_path = "yolov4-tiny.weights"
            config_path = "yolov4-tiny.cfg"

            model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            '''
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            '''

            layer_name = model.getLayerNames()
            layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

            video = cv2.VideoCapture(path)
            writer = None

            while True:
                (grabbed, image) = video.read()

                if not grabbed:
                    break

                image = imutils.resize(image, width=700)
                results = pedestrian_detection(image, model, layer_name,
                                               personidz=LABELS.index("person"))
                person=1

                for res in results:
                    cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)
                    cv2.putText(image, f'person {person}', (res[1][0],res[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    person+= 1
                    a=0

                cv2.putText(image, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
                cv2.putText(image, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)

                cv2.imshow("Human Detection from Video",image)
                a+=person-1
                #print(a)
                key = cv2.waitKey(1)

                if key & 0xFF == ord('q'):
                    break
            mbox.showinfo("TOTAL PERSONS:",a,parent=windowv)
            video.release()
            info1.config(text="                                                  ")
            # info2.config(text="                                                  ")
            if a>20:
                mbox.showinfo("Status",'Max. Human Detected is greater than MAX LIMIT.\n Region is Crowded.',parent=windowv)
            else:
                mbox.showinfo("Status","Max. Human Detected is in range of MAX LIMIT.\nRegion is not Crowded.",parent=windowv)
            info1.config(text="Status : Detection & Counting Completed")
            # info2.config(text="Max. Human Count : " + str(max_count2))
            cv2.destroyAllWindows()

    #for video-------------------------
        lbl1 = tk.Label(windowv, text="DETECT  FROM VIDEO", font=("Footlight MT Light", 50, "underline"), fg="black")
        lbl1.place(x=180, y=20)
        lbl2 = tk.Label(windowv, text="Selected Video", font=("Verdana", 30), fg="black")
        lbl2.place(x=80, y=200)
        path_text2 = tk.Text(windowv, height=1, width=37, font=("Arial", 30), bg="light grey", fg="black", borderwidth=2,relief="solid")
        path_text2.place(x=80, y=260)

        Button(windowv, text="SELECT", command=open_vid, cursor="hand2", font=("Arial", 20), bg="light blue", fg="black").place(x=220, y=350)
        Button(windowv, text="PREVIEW", command=prev_vid, cursor="hand2", font=("Arial", 20), bg="light blue", fg="black").place(x=410, y=350)
        Button(windowv, text="DETECT", command=det_vid, cursor="hand2", font=("Arial", 20), bg="light blue", fg="black").place(x=620, y=350)

        info1 = tk.Label(windowv, font=("Arial", 30), fg="gray")  # same way bg
        info1.place(x=100, y=440)
        # info2 = tk.Label(windowv, font=("Arial", 30), fg="gray")  # same way bg
        # info2.place(x=100, y=500)

        #function defined to exit from windowv section
        def exit_winv():
            if mbox.askokcancel("Exit", "Do you want to exit?", parent = windowv):
                windowv.destroy()
        windowv.protocol("WM_DELETE_WINDOW", exit_winv)
        

        # ---------------------------- camera section ------------------------------------------------------------
    def camera_option():
        # new window created for camera section
        windowc = tk.Tk()
        windowc.title("Human Detection from Camera")
        windowc.iconbitmap('Images/icon.ico')
        windowc.geometry('1000x700')

        # function defined to open the camera
        def open_cam():
            args = argsParser()

            info1.config(text="Status : Opening Camera...")
            # info2.config(text="                                                  ")
            mbox.showinfo("Status", "Opening Camera...Please Wait...", parent=windowc)
            # time.sleep(1)

            writer = None
            if args['output'] is not None:
                writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))
            if True:
                detectByCamera(writer)

        # function defined to detect from camera
        def detectByCamera(writer):
            video = cv2.VideoCapture(0)

            NMS_THRESHOLD=0.3
            MIN_CONFIDENCE=0.2

            def pedestrian_detection(image, model, layer_name,personidz=0):
                (H, W) = image.shape[:2]
                results = []

                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                             swapRB=True, crop=False)
                model.setInput(blob)
                layerOutputs = model.forward(layer_name)

                boxes = []
                centroids = []
                confidences = []

                for output in layerOutputs:
                    for detection in output:

                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        if classID == personidz and confidence > MIN_CONFIDENCE:

                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            boxes.append([x, y, int(width), int(height)])
                            centroids.append((centerX, centerY))
                            confidences.append(float(confidence))
                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
                # ensure at least one detection exists
                if len(idzs) > 0:
                    # loop over the indexes we are keeping
                    for i in idzs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        # update our results list to consist of the person
                        # prediction probability, bounding box coordinates,
                        # and the centroid
                        res = (confidences[i], (x, y, x + w, y + h), centroids[i])
                        results.append(res)

                #return the list of results
                return results

            labelsPath = "coco.names"
            LABELS = open(labelsPath).read().strip().split("\n")

            weights_path = "yolov4-tiny.weights"
            config_path = "yolov4-tiny.cfg"

            model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            '''
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            '''

            layer_name = model.getLayerNames()
            layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

            video = cv2.VideoCapture(0)
            writer = None

            while True:
                (grabbed, image) = video.read()

                if not grabbed:
                    break

                image = imutils.resize(image, width=700)
                results = pedestrian_detection(image, model, layer_name,
                                               personidz=LABELS.index("person"))
                person=1
                a=0

                for res in results:
                    cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)
                    cv2.putText(image, f'person {person}', (res[1][0],res[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    person+= 1
                cv2.putText(image, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
                cv2.putText(image, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
                cv2.imshow("Human Detection from WEb cam",image)
                a+=person-1
                #print(a)
                key = cv2.waitKey(1)

                if key & 0xFF == ord('q'):
                    break
            mbox.showinfo("TOTAL PERSONS:",a,parent=windowc)
            video.release()
            info1.config(text="                                                  ")
            # info2.config(text="                                                  ")
            if a>20:
                mbox.showinfo("Status",'Max. Human Detected is greater than MAX LIMIT.\n Region is Crowded.',parent=windowc)
            else:
                mbox.showinfo("Status","Max. Human Detected is in range of MAX LIMIT.\nRegion is not Crowded.",parent=windowc)
            info1.config(text="Status : Detection & Counting Completed")
            # info2.config(text="Max. Human Count : " + str(max_count2))
            cv2.destroyAllWindows()
            
        #for camera-------------------------------------------------
        lbl1 = tk.Label(windowc, text="DETECT  FROM CAMERA", font=("Footlight MT Light", 50, "underline"), fg="black")  # same way bg
        lbl1.place(x=180, y=20)

        Button(windowc, text="OPEN CAMERA", command=open_cam, cursor="hand2", font=("Arial", 20), bg="light blue", fg="black").place(x=370, y=200)

        info1 = tk.Label(windowc, font=("Arial", 30), fg="gray")  # same way bg
        info1.place(x=100, y=330)
        # info2 = tk.Label(windowc, font=("Arial", 30), fg="gray")  # same way bg
        # info2.place(x=100, y=390)

        # function defined to exit from the camera window
        def exit_winc():
            if mbox.askokcancel("Exit", "Do you want to exit?", parent = windowc):
                windowc.destroy()
        windowc.protocol("WM_DELETE_WINDOW", exit_winc)


            # options -----------------------------
    lbl1 = tk.Label(text="OPTIONS", font=("Footlight MT Light", 50, "underline"),fg="black")  # same way bg
    lbl1.place(x=340, y=20)

    # image on the main window
    pathi = "Images/img1.png"
    imgi = ImageTk.PhotoImage(Image.open(pathi))
    paneli = tk.Label(window1, image = imgi)
    paneli.place(x = 90, y = 110)

    # image on the main window
    pathv = "Images/img3.png"
    imgv = ImageTk.PhotoImage(Image.open(pathv))
    panelv = tk.Label(window1, image = imgv)
    panelv.place(x = 700, y = 260)# 720, 260

    # image on the main window
    pathc = "Images/img2.png"
    imgc = ImageTk.PhotoImage(Image.open(pathc))
    panelc = tk.Label(window1, image = imgc)
    panelc.place(x = 90, y = 415)

    # created button for all three option
    Button(window1, text="DETECT  FROM  IMAGE➡ ",command=image_option, cursor="hand2", font=("Arial",30), bg = "light blue", fg = "black").place(x = 350, y = 150)
    Button(window1, text="DETECT  FROM  VIDEO ➡",command=video_option, cursor="hand2", font=("Arial", 30), bg = "light blue", fg = "black").place(x = 110, y = 300) #90, 300
    Button(window1, text="DETECT  FROM  CAMERA ➡",command=camera_option, cursor="hand2", font=("Arial", 30), bg = "light blue", fg = "black").place(x = 350, y = 450)

    # function defined to exit from window1
    def exit_win1():
        if mbox.askokcancel("Exit", "Do you want to exit?"):
            window1.destroy()

    # created exit button
    Button(window1, text="❌ EXIT",command=exit_win1,  cursor="hand2", font=("Arial", 25), bg = "white", fg = "black").place(x = 440, y = 600)

    window1.protocol("WM_DELETE_WINDOW", exit_win1)
    window1.mainloop()
