#Core Packages
import streamlit as st

#EDA Packages
import pandas as pd 
import numpy as np

#Utilities
import os
<<<<<<< HEAD
import joblib
import hashlib
=======
#import joblib
>>>>>>> 3962a2454050ac20573dada78d2fc1ac58ae4dac

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
    		if password == "12345":
    			st.success("Welcome {}".format(username))

    			activity = st.selectbox("Activity", submenu)
    			if activity =="Crop Recommendation":
    				st.subheader("Crop Recommendation Engine")

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
