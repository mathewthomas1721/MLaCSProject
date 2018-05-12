import pandas as pd
from cvxpy import *
import numpy as np
import time
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from tt_split import *
import time
import sys
from sklearn.model_selection import GridSearchCV, PredefinedSplit,RandomizedSearchCV, ParameterGrid
from sklearn.metrics import mean_squared_error, make_scorer,log_loss
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc,roc_auc_score


FIG_PATH="../Figs/"
DATA_PATH="../Data/"
MODEL_PATH="./Models/"
Model_names = ["GBM","RanFor","Ada","LogReg","Bag","MLP","ExtraTree"]# prefixes for models
Model_names = ["RanFor","Bag","GBM"]# prefixes for models
##

def fit_numeric_members(season,X_train,y_train):
	model_dict={}
	print("Fitting Numeric Models For "+str(season)+" Season.")
	for prefix in Model_names:
		print("Fitting "+str(prefix)+" Model.")
		model= joblib.load(MODEL_PATH+prefix+"_num_"+str(season)+".pkl")
		model_dict[str(prefix)+"_"+"num"]= model.fit(X_train,y_train)
	return model_dict

def fit_categorical_members(season, X_train,y_train):
	model_dict={}
	print("Fitting Catgorical Models For "+str(season)+" Season.")
	for prefix in Model_names:
		print("Fitting "+str(prefix)+" Model.")
		model= joblib.load(MODEL_PATH+prefix+"_cat_"+str(season)+".pkl")
		model_dict[str(prefix)+"_"+"cat"]= model.fit(X_train,y_train)
	return model_dict

def fit_full_members(season,X_train,y_train):
	model_dict={}
	print("Fitting Full Models For "+str(season)+" Season.")
	for prefix in Model_names:
		print("Fitting "+str(prefix)+" Model.")
		model= joblib.load(MODEL_PATH+prefix+"_num_"+str(season)+".pkl")
		model_dict[str(prefix)+"_"+"num"]= model.fit(X_train,y_train)
	return model_dict

def plot_roc(y_true, y_hats,title,fname):

	k_probs = np.zeros((len(y_true),len(y_hats)))
	roc_auc=[]
	
	for i in range(len(y_hats)):
		k_probs[:,i]=y_hats[i].flatten()
		roc_auc.append(roc_auc_score(y_true, k_probs[:,i]))
	plt.figure()
	colors = ['aqua', 'darkorange', 'cornflowerblue','green']
	classes = ['numeric','categorical','full','aggregate']
	for k in range(len(y_hats)):
		fpr, tpr,thresh = roc_curve(y_true,k_probs[:,k])
		plt.plot(fpr, tpr, color=colors[k],label='{0} (area = {1:0.2f})'.format(classes[k], roc_auc[k]))
	plt.title(title)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(FIG_PATH+fname)
	#plt.show()
	plt.close()
	
def write_text(numeric,cat,full,agg,fname,season,model_type="Convex Learned Ensemble"):
	ostream = open(fname, "w")
	ostream.write("Results for "+str(model_type)+" in "+str(season)+"\n")
	ostream.write("\t numeric ensemble loss: "+str(numeric)+"\n")
	ostream.write("\t categorical ensemble loss: "+str(numeric)+"\n")
	ostream.write("\t full ensemble loss: "+str(numeric)+"\n")
	ostream.write("\t aggregate ensemble loss: "+str(numeric)+"\n")
	ostream.close()

def fit_ensemble_weights(f_X, y,equality=True):
	theta = Variable(f_X.shape[1],1)
	objective = Minimize(sum_entries(- (y*log(f_X*theta) +(1-y)*log(f_X*theta))))
	if equality:
		constraints = [sum_entries(theta)==1,theta>=0]
	else:
		constraints = [theta>=0]
	prob = Problem(objective,constraints)
	s = time.time()
	prob.solve()
	e = time.time()
	print("took "+str( (e-s)/60)+ " minutes.")
	optimal_weights = theta.value
	return optimal_weights

def make_prediction_matrix(model_dict,X):
	# input a dictionary of models and a set of training points
	# output a matrix of size n_instances (x.shape[0]) by n_models (len(model_dict.keys()))

	prediction_matrix = np.zeros((X.shape[0],len(model_dict.keys())))
	keylist = list(model_dict.keys())
	for k in range(len(model_dict.keys())):
		model = model_dict[keylist[k]]
		y_hat = model.predict_proba(X)[:,1]
		prediction_matrix[:,k]= y_hat
	return prediction_matrix


def plot_weights(theta,labels, title="",fname=""):
	plt.bar(np.arange(len(labels)),theta)
	plt.xticks(np.arange(len(labels)),labels)
	plt.title(title)
	plt.xlabel("Model Type")
	plt.ylabel("Ensemble Weights")
	ax = plt.subplot(111)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.savefig(fname)
	plt.close()


def make_ensemble(model_dict,X_train,X_test, y_train,y_test):
	# to fit an ensemble we need a matrix
	# f_X_train a matrix of size n_instances, n_models
	# y_train a vector of train samples
	# f_X_test and
	

	f_X_train = make_prediction_matrix(model_dict,X_train)
	f_X_test = make_prediction_matrix(model_dict,X_test)
	optimal_weights = fit_ensemble_weights(f_X_train,y_train)
	y_hat = np.dot(f_X_test,optimal_weights)
	LL = log_loss(y_test,y_hat)
	print(LL)
	return optimal_weights,LL,y_hat, f_X_train, f_X_test

def main():


	Seasons = ["2012","2013","2014","2015","2016","2017"]
	for season in Seasons:
		file_name = "../Data/RegularSeasonFeatures"+season+".csv"
		df = pd.read_csv(file_name,index_col=0)
		num_dex = df.columns.get_loc('wind2bkrate')-3
		X_train, X_test, y_train, y_test = train_test_split_season(df)

		# remove name features. 
		train_names = X_train[:,0:3]
		test_names = X_test[:,0:3]

		X_train = X_train[:,3:]
		X_test = X_test[:,3:]

		scaler = StandardScaler()
		scaler.fit(X_train[:,:num_dex])
		
		X_train[:,:num_dex]=scaler.transform(X_train[:,:num_dex])
		X_test[:,:num_dex]=scaler.transform(X_test[:,:num_dex])

		# fit the models
		numeric_models = fit_numeric_members(season, X_train[:,:num_dex],y_train)
		categorical_models = fit_categorical_members(season, X_train[:,num_dex:],y_train)
		full_models = fit_full_members(season, X_train,y_train)
		#make ensembles & evaluate

		#numerical
		num_weights, LL_n, yhat_n,trainmat_n, testmat_n = make_ensemble(numeric_models,X_train[:,:num_dex],X_test[:,:num_dex],y_train,y_test)
		plot_weights(num_weights,Model_names, "Ensemble Model Weights For "+str(season)+" Season\n Numeric Features Only","EnsembleWeights_n_"+str(season)+".pdf")
		
		#categorical 
		cat_weights, LL_c, yhat_c,trainmat_c, testmat_c = make_ensemble(categorical_models,X_train[:,num_dex:],X_test[:,num_dex:],y_train,y_test)
		plot_weights(cat_weights,Model_names, "Ensemble Model Weights For "+str(season)+" Season\n Categorical Features Only","EnsembleWeights_c_"+str(season)+".pdf")

		#full 
		full_weights, LL_f, yhat_f,trainmat_f, testmat_f = make_ensemble(full_models,X_train,X_test,y_train,y_test)
		plot_weights(full_weights,Model_names, "Ensemble Model Weights For "+str(season)+" Season\n Full Models","EnsembleWeights_f_"+str(season)+".pdf")
		


		#stacked ensemble
		big_train = np.hstack( (trainmat_n,trainmat_c,trainmat_f))
		big_test = np.hstack( (testmat_n,testmat_c,testmat_f))
		print("starting aggregates")
		agg_weights = fit_ensemble_weights(big_train,y_train)
		#plot_weights(agg_weights,Model_names, "Ensemble Model Weights For "+str(season)+" Season\n Aggregate Ensemble" ,"EnsembleWeights_a_"+str(season)+".pdf")
		yhat_a = np.dot(big_test,agg_weights)
		LL_a = log_loss(y_test,yhat_a)

		write_text(LL_n,LL_c,LL_f,LL_a,DATA_PATH+"EnsembleResults_"+str(season)+".txt",season,model_type="Convex Learned Ensemble")
		plot_roc(y_test,[yhat_n,yhat_c, yhat_f,yhat_a],"ROC Curves for "+str(season)+" Ensembles", "Ensemble_ROC_"+str(season)+".pdf")


if __name__ == '__main__':
	main()

