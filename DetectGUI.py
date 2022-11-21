import tkinter as tk
from tkinter import Label,Tk,filedialog,ttk,Button
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
import threading
import datetime
import imutils
import os

global im_org
global cap
mainWin=Tk()
mainWin.geometry("1200x600")
mainWin.title("Coins Detections")
Tabs=ttk.Notebook(mainWin)
Acc= ttk.Frame(Tabs)
FromImage= ttk.Frame(Tabs)
FromVideo= ttk.Frame(Tabs)
TitleLabel=Label(Acc,text="Projet detection pieces de monnaie",fg='red')
TitleLabel.config(font=('Helvetica bold',40))
TitleLabel.place(x=100,y=5)
textLabel=Label(Acc,text="Detecter les pieces dans une image",fg='red')
textLabel.config(font=('Helvetica bold',15))
textLabel.place(x=400,y=150)
text2Label=Label(Acc,text="Detecter les pieces depuis un webcam",fg='red')
text2Label.config(font=('Helvetica bold',15))
text2Label.place(x=400,y=200)
im = Image.open('AccImg.jpg').convert('RGB') 
im=im.resize((400,200))
AccImg=ImageTk.PhotoImage(im)
imageLabel=Label(Acc,image=AccImg)
imageLabel.place(x=400,y=240)
text2Label=Label(Acc,text="Par Ayoub El Kadmiri (MQL)",fg='white',bg='blue')
text2Label.config(font=('Helvetica bold',15))
text2Label.place(x=450,y=500)
def LoadImage():
    global im_org
    path=tk.filedialog.askopenfilename(filetypes=[("Image File",'*.*')])
    im_org=Image.open(path).convert('RGB') 
    im = Image.open(path).convert('RGB') 
    im=im.resize((400,350))
    imT=ImageTk.PhotoImage(im)
    label =Label(FromImage,image=imT)
    label.place(x=20,y=10)
    btn2=Button(FromImage,text='Detection',font =
               ('calibri', 10, 'bold', 'underline'),
                foreground = 'red',command=Detect)
    btn2.place(x=480,y=200)
    FromImage.mainloop()
def Detect():
    global im_org
    class_names=['10DH', '10c', '10cent', '1DH', '1c', '1e', '20c', '20cent', '2DH', '2c', '2e', '50c', '50cent', '5DH', '5c']
    model = tf.keras.models.load_model('ModelC')
    img = np.array(im_org) 
    img = img[:, :, ::-1].copy() 
    img_c=img.copy()
    img_c2=img.copy()
    img=cv2.medianBlur(img,5)
    img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3,3),0)
    circles = cv2.HoughCircles(img , cv2.HOUGH_GRADIENT,0.9,120,param1 = 50 , param2 = 30,minRadius=30 , maxRadius=140 )
    circles_rounded = np.uint16(np.around(circles))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in circles_rounded[0,:]:
        cv2.circle(img_c,(i[0],i[1]),i[2],(200,100,200),3)
        x=i[0]-i[2]-30
        y=i[1]-i[2]-20
        h=i[1]+i[2]+20
        w=i[0]+i[2]+10
        ROI = img_c2[y:h, x:w]
        ROI = cv2.resize(ROI, (128, 128))
        img_array = tf.keras.utils.img_to_array(ROI)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        # predictions = probability_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        k=class_names[np.argmax(predictions[0])]
        cv2.putText(img_c ,k, (i[0],i[1]), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
    im2 = Image.fromarray(img_c)
    im2=im2.resize((500,500))
    imT2=ImageTk.PhotoImage(im2)
    label2 =Label(FromImage,image=imT2)
    label2.place(x=650,y=5)
    labeltext2=Label(FromImage,text='Apres Prediction',fg='red',font=('Helvetica bold',20))
    labeltext2.place(x=800,y=530)
    FromImage.mainloop()
btn=Button(FromImage,text='choose image',font =
               ('calibri', 10, 'bold', 'underline'),
                foreground = 'red',command=LoadImage)
btn.place(x=150,y=430)

global label3
def StartRec():
    global cap
    label3 =Label(FromVideo)
    label3.place(x=50,y=40)
    model = tf.keras.models.load_model('ModelC')
    class_names=['10DH', '10cent Euro', '10cent DH', '1DH', '1cent Euro', '1 Euro', '20cent Euro', '20cent DH', '2DH', '2cent Euro', '2 Euro', '50 cent Euro', '50cent Dh', '5DH', '5cent Euro']
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        img_c=img.copy()
        img_c2=img.copy()
        img=cv2.medianBlur(img,5)
        img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3,3),0)
        circles = cv2.HoughCircles(img , cv2.HOUGH_GRADIENT,0.9,120,param1 = 50 , param2 = 30,minRadius=30 , maxRadius=140 )
        if circles is not None:
            circles_rounded = np.uint16(np.around(circles))
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in circles_rounded[0,:]:
                cv2.circle(img_c,(i[0],i[1]),i[2],(200,100,200),3)
                x=i[0]-i[2]-30
                y=i[1]-i[2]-20
                h=i[1]+i[2]+20
                w=i[0]+i[2]+10
                ROI = img_c2[y:h, x:w]
                if ROI is not None:
                    ROI = cv2.resize(ROI, (128, 128))
                    img_array = tf.keras.utils.img_to_array(ROI)
                    img_array = tf.expand_dims(img_array, 0)
                    predictions = model.predict(img_array)
                    # predictions = probability_model.predict(img_array)
                    
                    score = tf.nn.softmax(predictions[0])
                    k=class_names[np.argmax(predictions[0])]
                      
                    cv2.putText(img_c ,k, (i[0],i[1]), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
        im3 = Image.fromarray(img_c)
        im3=im3.resize((1100,500))
        imT3=ImageTk.PhotoImage(im3)
        label3.configure(image=imT3)
        FromVideo.update()
        q = cv2.waitKey(30) & 0xff
        if q==27:
            break
    cap.release()
def StopRec():
    global cap
    global label3
    cap.release()
    label3.destroy()
btn3=Button(FromVideo,text='Start',command=StartRec)
btn3.place(x=450,y=10)
btn4=Button(FromVideo,text='Stop',command=StopRec)
btn4.place(x=620,y=10)
Tabs.add(Acc, text='Acceuil')
Tabs.add(FromImage, text='Image')
Tabs.add(FromVideo, text='WebCam')
Tabs.pack(expand=1, fill='both')
mainWin.mainloop()
