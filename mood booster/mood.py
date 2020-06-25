def raise_frame(frame):
    frame.tkraise()

def song():
    if C=="angry":
        webbrowser.open_new_tab("https://soundcloud.com/")
    elif C=="happy":
        webbrowser.open_new_tab("https://soundcloud.com/")
    elif C=="sad":
        webbrowser.open_new_tab("https://soundcloud.com/")
    elif C=="suprise":
        webbrowser.open_new_tab("https://soundcloud.com/")
    else:
        webbrowser.open_new_tab("https://soundcloud.com/")

def video():
    if C=="angry":
        webbrowser.open_new_tab("https://soundcloud.com/")
    elif C=="happy":
        webbrowser.open_new_tab("https://soundcloud.com/")
    elif C=="sad":
        webbrowser.open_new_tab("https://soundcloud.com/")
    elif C=="suprise":
        webbrowser.open_new_tab("https://soundcloud.com/")
    else:
        webbrowser.open_new_tab("https://soundcloud.com/")

def coffee():
    if C=="angry":
        webbrowser.open_new_tab("https://soundcloud.com/")
    elif C=="happy":
        webbrowser.open_new_tab("https://soundcloud.com/")
    elif C=="sad":
        webbrowser.open_new_tab("https://soundcloud.com/")
    elif C=="suprise":
        webbrowser.open_new_tab("https://soundcloud.com/")
    else:
        webbrowser.open_new_tab("https://soundcloud.com/")

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def show_frame():

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)
    # loading models

     # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30))

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30))

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)

        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if emotion_text == 'angry':
            style.configure("BW.TLabel", foreground="black", background="red")
        elif emotion_text == 'sad':
            style.configure("BW.TLabel", foreground="black", background="blue")
        elif emotion_text == 'happy':
            style.configure("BW.TLabel", foreground="black", background="yellow")
        elif emotion_text == 'surprise':
            style.configure("BW.TLabel", foreground="black", background="orange")
        else:
            style.configure("BW.TLabel", foreground="black", background="green")

        mood['text']=emotion_text
        mood['style']="BW.TLabel"
        mood1['text']=emotion_text
        mood1['style']="BW.TLabel"
        global C
        C=emotion_text
        cv2.putText(cv2image, emotion_text, (face_coordinates[0]+20, face_coordinates[1]-20), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 0, 255),3)
        cv2.rectangle(cv2image, (face_coordinates[0], face_coordinates[1]),
                      (face_coordinates[0]+face_coordinates[3], face_coordinates[1]+face_coordinates[3]), (255, 0, 0), 4)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


import webbrowser
import PIL
from PIL import Image,ImageTk
import pytesseract
import cv2
from tkinter import *
from tkinter.ttk import *
import numpy as np
import cv2
import keras
from keras.models import load_model
from statistics import mode
global emotion_text
    # parameters for loading data and images
emotion_model_path = 'emotion_model.hdf5'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
C="Empty"
root = Tk()
f1 = Frame(root)
f2 = Frame(root)
for frame in (f1, f2):
    frame.grid(row=0, column=0, sticky='news')
style =Style()
style.configure("BW.TLabel", foreground="black", background="white")

lmain = Label(f1)
lmain.pack()

Label(f1, text='FRAME 1').pack(side=TOP)
Button(f1, text='webcam',command=show_frame).pack()
Button(f1, text='Close',command=root.destroy).pack()
Button(f1, text='Go to menu', command=lambda:raise_frame(f2)).pack()
Label(f1, text='MOOD : ').pack(side=LEFT)

mood= Label(f1, text="None",style="BW.TLabel")
mood.pack(side=LEFT)

Label(f2, text='FRAME 2').pack()
Label(f2, text='MOOD : ').pack(side=TOP,fill=X)
mood1= Label(f2, text="None",style="BW.TLabel")
mood1.pack(side=TOP,fill=X)

Button(f2, text='Go to webcam', command=lambda:raise_frame(f1)).pack()
Button(f2, text='Song', command=song).pack()
Button(f2, text='Video', command=video).pack()
Button(f2, text='Coffee', command=coffee).pack()
Button(f2, text='Close',command=root.destroy).pack()

raise_frame(f1)
root.mainloop()
cap.release()
cv2.destroyAllWindows()
