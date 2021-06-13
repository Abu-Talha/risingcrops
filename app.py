#Core Packages
import streamlit as st

#EDA Packages
import pandas as pd 
import numpy as np
import pickle

#Utilities
import os
import hashlib

#Data Viz Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

#Database
from managed_db import *

#Password
def  generate_hashes(password): 
	return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False

# Pkl Model - Random Forest
model = pickle.load(open('models/RandomForest.pkl','rb'))
	
feature_names_best = ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall', 'ph']

def main():
    """ Crop Recommendation Engine """
    st.title("Rising Crops")

    menu = ["About", "Sign Up", "Login"]
    submenu = ["Crop Recommendation" , "Disease Detection"]

    choice = st.sidebar.selectbox("", menu)
    if choice == "About":
    	st.subheader("Welcome to Rising Crops")
    	st.text("Modern Agricultural Practices")

    elif choice == "Login":
    	username = st.sidebar.text_input("Username")
    	password = st.sidebar.text_input("Password", type='password')
    	if st.sidebar.checkbox("Login"):
    		create_usertable()
    		hashed_pswd = generate_hashes(password)
    		result = login_user(username,verify_hashes(password,hashed_pswd))

    		#if password == "12345":
    		if result:
    			st.success("Welcome {}".format(username))

    			activity = st.selectbox("Activity", submenu)
    			if activity =="Crop Recommendation":
    				st.subheader("Crop Recommendation Engine")
    				df = pd.read_csv('data/crop_recommendation.csv')
    				
    				N = st.slider("N - Nitrogen Value" , 0,100)
    				P = st.slider("P - Phosphorus Value" , 0,100)
    				K = st.slider("K - Potassium Value" , 0,100)
    				ph = st.slider("PH of the soil" , 0,14)
    				temperature = st.number_input("Temperature (in C)", 0,100)
    				humidity = st.number_input("Humidity", 0,100)
    				rainfall = st.number_input("Rainfall (in mm)", 0,300)

    				data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    				my_prediction = model.predict(data)
    				final_prediction = my_prediction[0]

    				st.title(final_prediction)
    				st.write("Change the parameters as per your situation")


    			elif activity == "Disease Detection":
    				st.subheader("Disease Detection via Image Recognition")


    		else:
    			st.warning("Incorrect Username/Password")

    elif choice == "Sign Up":
    	new_username = st.text_input("User Name")
    	new_password = st.text_input("Password", type='password')

    	confirm_password = st.text_input("ConfirmPassword", type='password')
    	if new_password == confirm_password:
    		st.success("Password Confirmed")
    	else:
    		st.warning("Passwords are not the same")

    	if st.button("Submit"):
    		create_usertable()
    		hashed_new_password = generate_hashes(new_password)
    		add_userdata(new_username,hashed_new_password)
    		st.success("You have successfully created a new account")
    		st.info("Login to Get Started")

    		 			


if __name__ == '__main__':
    main()
