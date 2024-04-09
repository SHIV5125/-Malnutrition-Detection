from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog,QLabel
from PyQt5.QtGui import QPixmap
import tensorflow as tf

#from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from numpy import array
from keras import regularizers
import cv2
import matplotlib.pyplot as plt
from keras.models import model_from_json
import time 

import login
import home
import add
import error_log
import err_img
import sys
import MySQLdb

fname=""
fname1=""

db = MySQLdb.connect("localhost","root","Aditi0612!","malnutrition")
cursor = db.cursor()

class Login(QtWidgets.QMainWindow, login.Ui_UserLogin):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self) 
        self.pushButton.clicked.connect(self.log)
        self.pushButton_2.clicked.connect(self.addNew1)
        self.pushButton_3.clicked.connect(self.can)
        
    def log(self):
        i=0
        a=self.lineEdit.text()
        b=self.lineEdit_2.text()
        sql = "SELECT * FROM login WHERE name='%s' and password='%s'" % (a,b)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            for row in results:
                i=i+1
        except Exception as e:
           print (e)
        if i>=0:
            print ("login success")
            self.hide()
            self.home=home()
            self.home.show()
            
        else:
            print ("login failed")
            self.errlog=errlog()
            self.errlog.show()
                    
        db.close()
        
    def can(self):
        sys.exit()

    def addNew1(self):
        self.addNew=addNew()
        self.addNew.show()

class addNew(QtWidgets.QMainWindow, add.Ui_AdNewAdvertizer):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.save1)
        # self.pushButton_3.clicked.connect(self.can2)
        self.pushButton_2.clicked.connect(self.can2)

    def can2(self):
        sys.exit()
        
    def save1(self):
        name=self.lineEdit.text()
        email=self.lineEdit_2.text()
        contact=self.lineEdit_3.text()
        uname=self.lineEdit_4.text()
        pwd=self.lineEdit_5.text()
        sql = "INSERT INTO user(name, email, contact, username, password) VALUES ('%s', '%s', '%s', '%s', '%s' )" % (name,email,contact,uname,pwd)
        try:
                cursor.execute(sql)
                self.hide()
                db.commit()
        except:
                db.rollback()
                # self.erradd=erradd()
                self.erradd.show()
            

        db.close()
       

class home(QtWidgets.QMainWindow, home.Ui_Home):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.selimg)
        self.pushButton_4.clicked.connect(self.selimg2)
        self.pushButton_2.clicked.connect(self.seldir)
        self.pushButton_5.clicked.connect(self.ex)
        self.pushButton_3.clicked.connect(self.cnn)
        self.pushButton_6.clicked.connect(self.preproc)
        self.pushButton_7.clicked.connect(self.pred)
        

    def selimg(self):
        global fname
        # self.QFileDialog = QtWidgets.QWidget.QFileDialog(self)
        # self.QFileDialog.show()
        fname33 = QFileDialog.getOpenFileName(self)
        print(fname33)
        fname=fname33[0]
        print(fname)

        dim = (600, 600)
        img_to_show = cv2.imread(str(fname))
        resized = cv2.resize(img_to_show, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite("resized.jpg", resized)
        label = QLabel(self.label_5)
        pixmap = QPixmap("resized.jpg")
        label.setPixmap(pixmap)
        label.resize(pixmap.width(), pixmap.height())
        label.show()

    def selimg2(self):
        global fname1
        # self.QFileDialog = QtWidgets.QWidget.QFileDialog(self)
        # self.QFileDialog.show()
        fname55= QFileDialog.getOpenFileName(self)
        print(fname55[0])
        fname1=fname55[0]
        dim = (600, 600)
        img_to_show = cv2.imread(str(fname1))
        resized1 = cv2.resize(img_to_show, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite("resized1.jpg", resized1)
        label = QLabel(self.label_9)
        pixmap = QPixmap("resized1.jpg")
        label.setPixmap(pixmap)
        label.resize(pixmap.width(), pixmap.height())
        label.show()

    def seldir(self):
        # self.QFileDialog = QtWidgets.QWidget.QFileDialog(self)
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        print(folder)

    def cnn(self):
        # init the model
        model = Sequential()

        # add conv layers and pooling layers
        model.add(Convolution2D(32, 3,3, input_shape=(400,400,1),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3,3, input_shape=(400,400,1),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.5))  # to reduce overfitting

        model.add(Flatten())

        # Now two hidden(dense) layers:
        model.add(Dense(units= 4, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))

        model.add(Dropout(0.5))  # again for regularization

        model.add(Dense(units= 4, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))


        model.add(Dropout(0.5))  # last one lol

        model.add(Dense(units= 4, activation = 'relu',
                        kernel_regularizer=regularizers.l2(0.01)))

        # output layer
        model.add(Dense(units= 4, activation = 'sigmoid'))

        # Now copile it
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Now generate training and test sets from folders

        train_datagen =ImageDataGenerator(
            rescale = 1./255,
            shear_range = 0.2,
            zoom_range = 0.,
            horizontal_flip = False
            )

        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set =train_datagen.flow_from_directory("Datasets/training_set",
                                                       target_size = (400, 400),
                                                       color_mode='grayscale',
                                                       batch_size=1,
                                                       class_mode='categorical')

        test_set =test_datagen.flow_from_directory("Datasets/test_set",
                                                       target_size = (400, 400),
                                                  color_mode='grayscale',
                                                  batch_size=1,
                                                  class_mode='categorical')


        # finally, start training
        hiss=model.fit_generator(training_set,
                         epochs = 200,
                         validation_data = test_set,
                         validation_steps = 420)




        plt.figure(figsize=(15, 7))

        plt.subplot(1, 2,1)
        plt.plot(hiss.history['accuracy'], label='train')
        plt.plot(hiss.history['val_accuracy'], label='validation')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2,2)
        plt.plot(hiss.history['loss'], label='train')
        plt.plot(hiss.history['val_loss'], label='validation')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # saving the weights
        model.summary()

        model.save_weights("weights.hdf5", overwrite=True)

        # saving the model itself in json format:
        model_json = model.to_json()
        with open("model.json", "w") as model_file:
            model_file.write(model_json)
        print("Model has been saved.")

    def preproc(self):
        global fname
        global fname1
        if fname == "":
            self.errimg = errimg()
            self.errimg.show()
        else:
            path = [fname, fname1]
            for file1 in path:
                print(file1)
                filename = file1
                print ("file for processing", filename)
                image = cv2.imread(str(filename))
                cv2.imshow("Original Image", image)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imshow("Grayscale Conversion", gray)
                gray = cv2.bilateralFilter(gray, 11, 17, 17)
                cv2.imshow("Bilateral Filter", gray)
                edged = cv2.Canny(gray, 27, 40)
                cv2.imshow("Canny Edges", edged)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    
    def pred(self):
        global fname
        global fname1
        child_age = self.lineEdit.text()
        gender = self.lineEdit_4.text()
        weight = self.lineEdit_2.text()
        height = self.lineEdit_3.text()
        skin_pic = str(fname1)
        nails_pic = str(fname)


        skin = ""
        nail = ""
        skin1 = ""
        nail1 = ""
        nutriw = ""
        nutrih = ""
        histarray = {'healthy_nails':0, 'unhealthy_nails':0, 'healthy_skin': 0, 'unhealthy_skin': 0}

        def load_model():
            try:
                json_file = open('model.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)
                model.load_weights("weights.hdf5")
                print("Model successfully loaded from disk.")

                # compile again
                model.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
                return model
            except Exception as e:
                print(e)
                print("""Model not found. Please train the CNN by running the script """)
                return None


        def update(histarray2):
            global histarray
            histarray = histarray2


        def realtime(pic):
            classes = ['healthy_nails', 'unhealthy_nails', 'healthy_skin', 'unhealthy_skin']

            frame = cv2.imread(pic)
            frame = cv2.resize(frame, (400, 400))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame.reshape((1,)+frame.shape)
            frame = frame.reshape(frame.shape+(1,))
            test_datagen = ImageDataGenerator(rescale=1./255)
            m = test_datagen.flow(frame,batch_size=1)
            y_pred = model.predict_generator(m,1)
            histarray2 = {'healthy_nails': y_pred[0][0], 'unhealthy_nails': y_pred[0][1], 'healthy_skin': y_pred[0][2], 'unhealthy_skin': y_pred[0][3]}
            update(histarray2)
            pred = classes[list(y_pred[0]).index(y_pred[0].max())]
            return pred


        model = load_model()
        print ("skin Pic=", skin_pic)
        print ("Nails Pic=", nails_pic)
        skin = realtime(skin_pic)
        nails = realtime(nails_pic)


        wt = 0
        ht = 0
        wt1 = 0
        ht1 = 0
        sql = "SELECT * FROM nutrifact where age='%f' and gender='%s'" %(float(child_age), gender)
        print ("sql=", sql)

        results2 = ""
        try:
            cursor.execute(sql)
            results2 = cursor.fetchall()
        except Exception as e:
            print(e)


        results2 = ""
        for row in results2:
            wt = int(row[2])
            ht = int(row[1])

        wt1 = float(weight)-wt
        ht1 = float(height)-ht

        if wt1 <25:
            nutriw = "Weight is less of child consult with doctor for nutrition"
            self.textEdit.append(nutriw)
        else:
            nutriw = "Weight is good and according to nutrition parameters"
            self.textEdit.append(nutriw)

        if ht1 < 130:
            nutrih = "Height is less of child consult with doctor for nutrition"
            self.textEdit.append(nutrih)
        else:
            nutrih = "Height is good and according to nutrition parameters"
            self.textEdit.append(nutrih)

        if 'unhealthy_skin' in skin:
            skin1 = "Need Skin care or may be skin nutrition is less"
            self.textEdit.append(skin1)
        else:
            skin1 = "Skin care is good"
            self.textEdit.append(skin1)

        if 'healthy_nails' in nails:
            nail1 = "Nails and nutrition seems good"
            self.textEdit.append(nail1)
        else:
            nail1 = "Need care of nails "
            self.textEdit.append(nail1)



    def ex(self):
        sys.exit()
 
        

    
class errlog(QtWidgets.QMainWindow, error_log.Ui_Error):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.ok1)
    def ok1(self):
        self.hide()

class errimg(QtWidgets.QMainWindow, err_img.Ui_Error):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.ok1)
    def ok1(self):
        self.hide()



def main():
    app = QtWidgets.QApplication(sys.argv)  
    form = Login()                 
    form.show()                         
    app.exec_()                         


if __name__ == '__main__':              
    main()                             
