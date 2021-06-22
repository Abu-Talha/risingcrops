#Core Packages
import streamlit as st
st.beta_set_page_config(page_title='Rising Crops')
#EDA Packages
import pandas as pd 
import numpy as np
import pickle
import joblib
import torch
from google.cloud import storage
#Utilities
import os
import hashlib
from sklearn import preprocessing
#Data Viz Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
#Disease Detection Packages
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchsummary import summary


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

season_dict = {"kharif":0,"rabi":1,"zaid":2}

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key

def get_season_value(val):
	season_dict = {"kharif":0,"rabi":1,"zaid":2}
	for key,value in season_dict.items():
		if val == key:
			return value

def load_model(model):
	loaded_model = joblib.load(open(os.path.join(model),"rb"))
	return loaded_model

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
    				df = pd.read_csv('https://www.kaggle.com/abutalhai/crop-recommendation1csv/download', error_bad_lines=False) #data/crop_recommendation1.csv
    				#category_col = ['season']
    				#labelEncoder = preprocessing.LabelEncoder()
    				#mapping_dict={}
    				#for col in category_col:
    					#df[col] = labelEncoder.fit_transform(df[col])
    					#le_name_mapping=dict(zip(labelEncoder.classes_,labelEncoder.transform(labelEncoder.classes_)))
    					#mapping_dict[col]=le_name_mapping


    				
    				N = st.slider("N - Nitrogen Value" , 0,100)
    				P = st.slider("P - Phosphorus Value" , 0,100)
    				K = st.slider("K - Potassium Value" , 0,100)
    				ph = st.slider("PH of the soil" , 0.0,14.0)
    				humidity = st.number_input("Humidity",0.0,100.0)
    				temperature = st.number_input("Temperature (in C)", 0.0,100.0)
    				season = st.radio("Season: 0 - kharif | 1 - rabi | 2 - zaid",(0,1,2))
    				rainfall = st.number_input("Rainfall (in mm)", 0.0,300.0)
				
				
    				data = np.array([[N, P, K, temperature, humidity, ph, season,rainfall]])
    				my_prediction = model.predict(data)
    				final_prediction = my_prediction[0]

    				st.title(final_prediction)
    				st.write("Change the parameters as per your situation")


    			elif activity == "Disease Detection":
    				st.subheader("Disease Detection via Image Recognition")

    				data_dir = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
    				train_dir = data_dir + "/train"
    				valid_dir = data_dir + "/valid"
    				diseases = os.listdir(train_dir)
    				train = ImageFolder(train_dir, transform=transforms.ToTensor())
    				valid = ImageFolder(valid_dir, transform=transforms.ToTensor())
    				batch_size=32
    				train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    				valid_dl = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)
    				device = torch.device("cpu")
    				
    				def to_device(data,device):
    					if isinstance(data,(list,tuple)):
    						return [to_device(x,device) for x in data]
    					return data.to(device, non_blocking=True)
    				
    				class DeviceDataLoader():
    					def __init__(self,dl,device):
    						self.dl=dl
    						self.device=device
    					
    					def __iter__(self):
    						for b in self.dl:
    							yield to_device(b, self.device)
    					
    					def __len__(self):
    						return len(self.dl)
    				
    				train_dl = DeviceDataLoader(train_dl,device)
    				valid_dl = DeviceDataLoader(valid_dl, device)
    				
    				class SimpleResidualBlock(nn.Module):
    					def __init__(self):
    						super().__init__()
    						self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, stride=1, padding=1)
    						self.relu1 = nn.ReLU()
    						self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, stride=1, padding=1)
    						self.relu2 = nn.ReLU()
    					
    					def forward(self, x):
    						out = self.conv1(x)
    						out = self.relu1(out)
    						out = self.conv2(out)
    						return self.relu2(out) + x 
    					
    				def accuracy(output,labels):
    					_, preds = torch.max(outputs, dim=1)
    					return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    				class ImageClassificationBase(nn.Module):
    					def training_step(self, batch):
    						images, labels = batch
    						out = self(images)
    						loss = F.cross_entropy(out,labels)
    						return loss

    					def validation_step(self, batch):
    						images, labels = batch
    						out = self(images)
    						loss = F.cross_entropy(out, labels)
    						acc = accuracy(out, labels)
    						return {"val_loss": loss.detach(), "val_accuracy": acc}

    					def validation_epoch_end(self,outputs):
    						batch_losses = [x["val_loss"] for x in outputs]
    						batch_accuracy = [x["val_accuracy"] for x in outputs]
    						epoch_loss = torch.stack(batch_losses).mean()
    						epoch_accuracy = torch.stack(batch_accuracy).mean()
    						return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}

    					def epoch_end(self,epoch,result):
    						print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

    				def ConvBlock(in_channels, out_channels, pool=False):
    					layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    					if pool:
    						layers.append(nn.MaxPool2d(4))
    					return nn.Sequential(*layers)

    				class ResNet9(ImageClassificationBase):
    					def __init__(self, in_channels, num_diseases):
    						super().__init__()

    						self.conv1 = ConvBlock(in_channels, 64)
    						self.conv2 = ConvBlock(64, 128, pool=True)
    						self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
    						self.conv3 = ConvBlock(128, 256, pool=True)
    						self.conv4 = ConvBlock(256, 512, pool=True)
    						self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
    						self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases))

    					def forward(self,xb):
    						out = self.conv1(xb)
    						out = self.conv2(out)
    						out = self.res1(out) + out 
    						out = self.conv3(out)
    						out = self.conv4(out)
    						out = self.res2(out) + out
    						out = self.classifier(out)
    						return out
    				model_d = to_device(ResNet9(3, len(train.classes)), device)
    				@torch.no_grad()
    				def evaluate(model_d, val_loader):
    					model_d.eval()
    					outputs = [model_d.validation_step(batch) for batch in val_loader]
    					return model_d.validation_epoch_end(outputs)

    				def get_lr(optimizer):
    					for param_group in optimizer.param_groups:
    						return param_group['lr']

    				def fit_OneCycle(epochs, max_lr, model_d, train_loader, val_loader, weight_decay=0,grad_clip=None, opt_func=torch.optim.SGD):
    					torch.cpu.empty_cache()
    					history=[]
    					optimizer = opt_func(model_d.parameters(), max_lr, weight_decay=weight_decay)
    					sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    					for epoch in range(epochs):
    						model_d.train()
    						train_losses = []
    						lrs = []
    						for batch in train_loader:
    							loss = model_d.training_step(batch)
    							train_losses.append(loss)
    							loss.backward()
    							if grad_clip:
    								nn.utils.clip_grad_value_(model_d.parameters(), grad_clip)
    							optimizer.step()
    							optimizer.zero_grad()
    							lrs.append(get_lr(optimizer))
    							sched.step()
    						result = evaluate(model_d, val_loader)
    						result['train_loss'] = torch.stack(train_losses).mean().item()
    						result['lrs'] = lrs
    						model_d.epoch_end(epoch,result)
    						history.append(result)
    					return history

    				test_dir = ('data/test')
    				test = ImageFolder(test_dir, transform=transforms.ToTensor())
    				test_images = sorted(os.listdir(test_dir + '/test'))

    				def predict_image(img, model_d):
    					xb = to_device(img.unsqueeze(0), device)
    					yb = model_d(xb)
    					_, preds = torch.max(yb, dim=1)
    					return train.classes[preds[0].item()]

    				img = st.file_uploader("Pick a File")
    				img,label = test[0]
    				plt.imshow(img.permute(1,2,0))
    				st.write("Disease Predicted:", predict_image(img,model_d))






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
