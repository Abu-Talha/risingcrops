#Core Packages
import streamlit as st
st.beta_set_page_config(page_title='Rising Crops')
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

#Weather Data
import pytz
import pyowm
from matplotlib import dates
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
owm = pyowm.OWM('715769715b24cccf5b2384e4c515e5dd')
mgr=owm.weather_manager()

degree_sign= u'\N{DEGREE SIGN}'

season_dict = {"Kharif":0,"Rabi":1,"Zaid":2}

# Pkl Model - Random Forest
model = pickle.load(open('models/RandomForest1.pkl','rb'))
	
feature_names_best = ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall', 'ph','season']

def main():
    """ Crop Recommendation Engine """
    st.title("Rising Crops")

    menu = ["About", "Sign Up", "Login"]
    submenu = ["Crop Recommendation" , "Disease Detection" , "Weather Forecast"]

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
                    df = pd.read_csv('data/crop_recommendation1.csv')
                    N = st.slider("N - Nitrogen Value" , 0,100)
                    P = st.slider("P - Phosphorus Value" , 0,100)
                    K = st.slider("K - Potassium Value" , 0,100)
                    ph = st.slider("PH of the soil" , 0.0,14.0)
                    temperature = st.number_input("Temperature (in C)", 0.0,100.0)
                    humidity = st.number_input("Humidity", 0.0,100.0)
                    rainfall = st.number_input("Rainfall (in mm)", 0.0,300.0)
                    data = np.array([[N, P, K, temperature, humidity, ph, season,rainfall]])
                    my_prediction = model.predict(data)
                    final_prediction = my_prediction[0]

                    st.title(final_prediction)
                    st.write("Change the parameters as per your situation")

                elif activity == "Disease Detection":
                    st.subheader("Disease Detection via Image Recognition")

                elif activity == "Weather Forecast":
                    st.subheader("5 day Weather Forecast")

                    place=st.text_input("NAME OF THE CITY:","")

                    if place == None:
                        st.write("Input a CITY!")

                    unit=st.selectbox("Select Temperature Unit",("Celsius","Fahrenheit"))
                    g_type=st.selectbox("Select Graph Type",("Line Graph","Bar Graph"))
                    if unit == 'Celsius':
                        unit_c = 'celsius'
                    else:
                        unit_c = 'fahrenheit'

                    def get_temperature():
                        days = []
                        dates = []
                        temp_min = []
    					temp_max = []
    					forecaster = mgr.forecast_at_place(place, '3h')
    					forecast=forecaster.forecast
    					for weather in forecast:
    						day=datetime.utcfromtimestamp(weather.reference_time())
    						#day = gmt_to_eastern(weather.reference_time())
    						date = day.date()
    						if date not in dates:
    							dates.append(date)
    							temp_min.append(None)
    							temp_max.append(None)
    							days.append(date)
    						temperature = weather.temperature(unit_c)['temp']
    						if not temp_min[-1] or temperature < temp_min[-1]:
    							temp_min[-1] = temperature
    						if not temp_max[-1] or temperature > temp_max[-1]:
    							temp_max[-1] = temperature
    					return(days, temp_min, temp_max)

    				def init_plot():
    					plt.figure('PyOWM Weather', figsize=(5,4))
    					plt.xlabel('Day')
    					plt.ylabel(f'Temperature ({degree_sign}F)')
    					plt.title('Weekly Forecast')

    				def plot_temperatures(days, temp_min, temp_max):
    				# days = dates.date2num(days)
    					fig = go.Figure(
    						data=[
    							go.Bar(name='minimum temperatures', x=days, y=temp_min),
    							go.Bar(name='maximum temperatures', x=days, y=temp_max)
    						]
    					)
    					fig.update_layout(barmode='group')
    					return fig

    				def plot_temperatures_line(days, temp_min, temp_max):
    					fig = go.Figure()
    					fig.add_trace(go.Scatter(x=days, y=temp_min, name='minimum temperatures'))
    					fig.add_trace(go.Scatter(x=days, y=temp_max, name='maximimum temperatures'))
    					return fig

    				def label_xaxis(days):
    					plt.xticks(days)
    					axes = plt.gca()
    					xaxis_format = dates.DateFormatter('%m/%d')
    					axes.xaxis.set_major_formatter(xaxis_format)

    				def draw_bar_chart():
    					days, temp_min, temp_max = get_temperature()
    					fig = plot_temperatures(days, temp_min, temp_max)
    					# write_temperatures_on_bar_chart(bar_min, bar_max)
    					st.plotly_chart(fig)
    					st.title("Minimum and Maximum Temperatures")
    					for i in range (0,5):
    						st.write("### ",temp_min[i],degree_sign,' --- ',temp_max[i],degree_sign)
    				def draw_line_chart():
    					days, temp_min, temp_max = get_temperature()
    					fig = plot_temperatures_line(days, temp_min, temp_max)
    					st.plotly_chart(fig)
    					st.title("Minimum and Maximum Temperatures")
    					for i in range (0,5):
    						st.write("### ",temp_min[i],degree_sign,' --- ',temp_max[i],degree_sign)

    				def other_weather_updates():
    					forecaster = mgr.forecast_at_place(place, '3h')
    					st.title("Impending Temperature Changes :")
    					if forecaster.will_have_fog():
    						st.write("### FOG Alert!")
    					if forecaster.will_have_rain():
    						st.write("### Rain Alert")
    					if forecaster.will_have_storm():
    						st.write("### Storm Alert!")
    					if forecaster.will_have_snow():
    						st.write("### Snow Alert!")
    					if forecaster.will_have_tornado():
    						st.write("### Tornado Alert!")
    					if forecaster.will_have_hurricane():
    						st.write("### Hurricane Alert!")
    					if forecaster.will_have_clouds():
    						st.write("### Cloudy Skies")    
    					if forecaster.will_have_clear():
    						st.write("### Clear Weather!")

    				def cloud_and_wind():
    					obs=mgr.weather_at_place(place)
    					weather=obs.weather
    					cloud_cov=weather.clouds
    					winds=weather.wind()['speed']
    					st.title("Cloud coverage and wind speed")
    					st.write('### The current cloud coverage for',place,'is',cloud_cov,'%')
    					st.write('### The current wind speed for',place, 'is',winds,'mph')

    				def sunrise_and_sunset():
    					obs=mgr.weather_at_place(place)
    					weather=obs.weather
    					st.title("Sunrise and Sunset Times :")
    					india = pytz.timezone("Asia/Kolkata")
    					ss=weather.sunset_time(timeformat='iso')
    					sr=weather.sunrise_time(timeformat='iso')  
    					st.write("### Sunrise time in",place,"is",sr)
    					st.write("### Sunset time in",place,"is",ss)

    				def updates():
    					other_weather_updates()
    					cloud_and_wind()
    					sunrise_and_sunset()
    				if st.button("SUBMIT"):
    					if g_type == 'Line Graph':
    						draw_line_chart()
    					else:
    						draw_bar_chart()
    					updates()





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
