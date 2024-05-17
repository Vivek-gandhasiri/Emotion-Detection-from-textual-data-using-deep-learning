from random import random
from xml.parsers.expat import model
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle

import warnings
import numpy as np

import sqlite3
from googletrans import Translator
import warnings

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import pandas as pd
import sqlite3
import random

import smtplib 
from email.message import EmailMessage
from datetime import datetime

from tensorflow.keras import backend as K


from topic_modelling import Topic_modeling

app = Flask(__name__)

#Reading data
translator = Translator()


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



model1 = load_model('model.h5',custom_objects={'f1_score' : f1_m, 'precision_score' : precision_m, 'recall_score' : recall_m}, compile=False)

cv1=pickle.load(open('transform2.pkl','rb'))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signin.html")


@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        
        message = request.form['message']
        translations = translator.translate(message, dest='en')
        message =  translations.text
        data = [message]
        vect = cv1.texts_to_sequences(data)
        vect = pad_sequences(vect)
        k=np.zeros((1,300))
        k[0,-vect.shape[1]:]=vect
        my_prediction = model1.predict_classes(np.array(k))
        df = pd.DataFrame({'sentence':data})
        t,word = Topic_modeling(df)
        return render_template('result.html',prediction = my_prediction[0],message=message,to = t, wo = word)


@app.route('/index')
def index():
	return render_template('index.html')


@app.route('/notebook')
def notebook():
	return render_template('Notebook.html')

@app.route('/about')
def about():
	return render_template('about.html')

if __name__ == '__main__':
	app.run(debug=False)
