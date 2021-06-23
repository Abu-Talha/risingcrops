#Database testing ---> python -m pytest
import managed_db

def database():
	managed_db.create_usertable()
	sample_data = [
	('abu','talha'),
	('adithya','raj')
	]

	managed_db.add_userdata(sample_data)

def test_database():
	assert len(list(managed_db.view_all_users())) == 2

#Model testing
import pickle
import pandas as pd
from sklearn import preprocessing
PATH = 'https://www.kaggle.com/abutalhai/crop-recommendation1csv/download'
df = pd.read_csv(PATH)
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'season', 'rainfall']]
target = df['label']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
load_model = pickle.load(open('models/RandomForest1.pkl','rb'))
acc_score = load_model.score(Xtest,Ytest)
data = np.array([[N, P, K, temperature, humidity, ph, season,rainfall]])
my_prediction = model.predict(data)
final_prediction = my_prediction[0]

def test_random_forest():

	assert acc_score == 0.9022727272727272

def test_prediction():
	(N,P,K,temperature,humidity,ph,season,rainfall)=(92,42,43,20.8,82,6.5,0,202)
	assert final_prediction == 'rice'
