import csv
import time
import tkinter as tk
import cv2, os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime

#GUI
window = tk.Tk()
window.title("Attendance Monitoring System")
window.geometry('1080x720')
window.iconbitmap("DTC.ico")
load = Image.open('DTC1N.jpeg')
load2 = Image.open('Logo-DTC-1.png')
render = ImageTk.PhotoImage(load)
render2 = ImageTk.PhotoImage(load2)
img = tk.Label(window,image = render)
img.place(x=0,y=0)
img2 = tk.Label(window,image = render2)
img2.place(x=480,y=3)
window.configure(background='white')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


#Heading and Credits
message = tk.Label(window, text="ATTENDANCE MONITORING SYSTEM USING FACE RECOGNITION", bg='blue', fg='white', width=60, height=2, font=('ariel',20,'bold'))
message.place(x=250, y=160)
message3 = tk.Label(window, text="Created By:- Vaibhav Jha", bg='light green', fg='black', width=130, height=1, font=('times',15,'bold italic'))
message3.place(x=50, y=720)


#Labels and Entry
lbl=tk.Label(window, text="Enter ID", width=20, height =2, fg='red', bg='white', font=('ariel',15,'bold'))
lbl.place(x=250, y=250)
txt=tk.Entry(window, width=20, bg="white", fg="red", font=('ariel', 25, 'bold'))
txt.place(x=570, y=250)
lbl2=tk.Label(window, text="Enter Name", width=20, height=2, fg='red', bg='white', font=('ariel',15,'bold'))
lbl2.place(x=250, y=330)
txt2=tk.Entry(window, width=20, bg="white", fg="red", font=('ariel', 25, 'bold'))
txt2.place(x=570, y=330)
lbl3=tk.Label(window, text="Message", width=20, height=2, fg='red', bg='white', font=('ariel',15,'bold'))
lbl3.place(x=250, y=410)
message1=tk.Label(window, text="", width=30, height=2, fg='red', bg='white', font=('ariel',15,'bold'))
message1.place(x=570, y=410)

#For Attendance
lbl4=tk.Label(window, text="Attendance", width=20, height=2, fg='red', bg='white', font=('ariel',15,'bold'))
lbl4.place(x=250, y=600)
message2=tk.Label(window, text="", width=45, height=4, fg='red', bg='white', font=('ariel',10,'bold'))
message2.place(x=570, y=600)
lbl5=tk.Label(window, text="Subject", width=20, height=1, fg='red', bg='white', font=('ariel',15,'bold'))
lbl5.place(x=1050, y=600)
txt3=tk.Entry(window, width=22, bg="white", fg="red", font=('ariel', 15, 'bold'))
txt3.place(x=1050, y=640)


#Clear Functions
def clear():
    txt.delete(0,'end')
    res=""
    message1.configure(text=res)

def clear2():
    txt2.delete(0,'end')
    res=""
    message1.configure(text=res)

def clear3():
    txt3.delete(0,'end')
    res=""
    message2.configure(text=res)



#Functions
def TakeImage():
    Id=(txt.get())
    name=(txt2.get())
    if(Id.isdigit() and name.isalpha()):
        cam=cv2.VideoCapture(0)
        harcascadePath="haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img=cam.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces= detector.detectMultiScale(gray,1.3,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                sampleNum=sampleNum+1
                cv2.imwrite("TrainingImages\ "+name +'.'+ Id + '.' +str(sampleNum) + ".jpg", gray[y:y+h,x:x+h])
                cv2.imshow('Frame',img)
            if cv2.waitKey(10) & 0xFF == ord('g'):
                            break
            elif sampleNum>59:
                break
        cam.release()
        cv2.destroyAllWindows()
        res="Saved for ID:" + Id +" Name:"+ name
        row= [Id,name]
        with open('StudentDetails\studentDetails.csv','a+') as csvFile:
            writer =csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=res)
    else:
        if(Id.isdigit()):
            res= "Enter Alphabetical Name"
            message1.configure(text =res)
        if(name.isalpha()):
            res="Enter numeric Id"
            message1.configure(text =res)


def TrainImg():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath="haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    face, Id = getImagesAndLabels("TrainingImages")
    recognizer.train(face, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained!"
    message1.configure(text = res)


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    Ids=[]

    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces,Ids


def TAttendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails\studentDetails.csv")
    cam=cv2.VideoCapture(0)
    font=cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id','Name','Date','Time']
    attendance=pd.DataFrame(columns = col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+h])
            if(conf<50):
                ts = time.time()
                date= datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
            else:
                Id='Unknown'
                tt=str(Id)
            if(conf>75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile)+".jpg",im[y:y+h,x:x+w])
            cv2.putText(im,str(tt),(x,y+h),font,1,(255,255,255),2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im',im)
        if(cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    subj = (txt3.get())
    fileName="Attendance\Attendance_"+subj+"_"+date+"_"+Hour+"_"+Minute+"_"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    message2.configure(text=res)



#Buttons in main window
clrButton=tk.Button(window, text="Clear",command=clear, width=20, height=2, fg='red', activebackground="green", font=('ariel',10,'bold'))
clrButton.place(x=1000, y=250)
clrButton2=tk.Button(window, text="Clear",command=clear2, width=20, height=2, fg='red', activebackground="green", font=('ariel',10,'bold'))
clrButton2.place(x=1000, y=330)
clrButton3=tk.Button(window, text="Clear",command=clear3, width=20, height=2, fg='red', activebackground="green", font=('ariel',10,'bold'))
clrButton3.place(x=1000, y=410)

takeImg=tk.Button(window, text="Take Image",command=TakeImage, width=20, height=2, fg='Blue', bg='grey', activebackground="red", font=('ariel',15,'bold'))
takeImg.place(x=200, y=500)
trainImg=tk.Button(window, text="Train Image",command=TrainImg, width=20, height=2, fg='Blue', bg='grey', activebackground="red", font=('ariel',15,'bold'))
trainImg.place(x=500, y=500)
trackImg=tk.Button(window, text="Take Attendance",command=TAttendance, width=20, height=2, fg='Blue', bg='grey', activebackground="red", font=('ariel',15,'bold'))
trackImg.place(x=800, y=500)
quitwin=tk.Button(window, text="Quit",command=window.destroy, width=20, height=2, fg='Blue', bg='grey', activebackground="red", font=('ariel',15,'bold'))
quitwin.place(x=1100, y=500)


window.mainloop()