import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tt_split import *
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import mean_squared_error, make_scorer,log_loss
FIG_PATH="../Figs/FigsNew/"
def evaluate_numeric(model,X_train, X_test, y_train,y_test,numdex,scale=True):
	X_train = X_train[:,0:numdex]
	X_test = X_test[:,0:numdex]

	if scale:
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train=scaler.transform(X_train)
		X_test=scaler.transform(X_test)
	model.fit(X_train,y_train)
	y_hat = model.predict_proba(X_test)
	#joblib.dump(model, name)
	ll = log_loss(y_test,y_hat)
	return y_hat,ll

def evaluate_categorical(model, X_train, X_test, y_train, y_test, catdex):
	X_train =X_train[:,catdex:]
	X_test = X_test[:,catdex:]
	model.fit(X_train,y_train)
	y_hat = model.predict_proba(X_test)
	ll = log_loss(y_test,y_hat)
	#joblib.dump(model,name)
	return y_hat,ll

def evaluate_full(model, X_train, X_test, y_train, y_test, splitdex,scale=True):

	if scale:
		scaler = StandardScaler()
		scaler.fit(X_train[:,0:splitdex])
		X_train[:,0:splitdex]=scaler.transform(X_train[:,0:splitdex])
		X_test[:,0:splitdex]=scaler.transform(X_test[:,0:splitdex])
	model.fit(X_train,y_train)
	#joblib.dump(model,name)
	y_hat = model.predict_proba(X_test)
	ll = log_loss(y_test,y_hat)
	return y_hat,ll

def plot_roc(y_true, y_hats,title,fname):

	k_probs = np.zeros((len(y_true),len(y_hats)))
	roc_auc=[]

	for i in range(len(y_hats)):
		k_probs[:,i]=y_hats[i][:,1]
		roc_auc.append(roc_auc_score(y_true, k_probs[:,i]))
	plt.figure()
	colors = ['aqua', 'darkorange', 'cornflowerblue']
	classes = ['numeric','categorical','full']
	for k in range(len(y_hats)):
		fpr, tpr,thresh = roc_curve(y_true,k_probs[:,k])
		plt.plot(fpr, tpr, color=colors[k],label='{0} (area = {1:0.5f})'.format(classes[k], roc_auc[k]))
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

def write_text(numeric,cat,full,fname,season,model_type):
	ostream = open(fname, "w")
	ostream.write("Results for "+str(model_type)+" in "+str(season)+"\n")
	ostream.write("\t numeric log loss: "+str(numeric)+"\n")
	ostream.write("\t categorical log loss: "+str(cat)+"\n")
	ostream.write("\t full model loss: "+str(full)+"\n")
	ostream.close()



mypath = "Models"
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#onlyfiles.sort()
#print(onlyfiles)
for season in ["2012","2013","2014","2015","2016","2017"]:
    #files = [file for file in onlyfiles if season in file]
    #print(files)
    #What do i need? Diags for each type of classifier
    for algo in ["GBM"]:#,"Ada","ExtraTree","LogReg","MLP","RanFor"]:
        print(algo + season)
        numeric_model = joblib.load(mypath + "/" +  algo + "_full_" + season + ".pkl")
        cat_model = joblib.load(mypath + "/" + algo + "_cat_" + season + ".pkl")
        full_model = joblib.load(mypath + "/" + algo + "_num_" + season + ".pkl")


        #type = file[-10:-9]

        #model = joblib.load(mypath + "/" + file)
        file_name = "../Data/RegularSeasonFeatures"+str(season)+".csv"
        df = pd.read_csv(file_name,index_col=0)
        X_train, X_test, y_train, y_test = train_test_split_season(df)


        train_names = X_train[:,0:3]
        test_names = X_test[:,0:3]

        X_train = X_train[:,3:]
        X_test = X_test[:,3:]

        num_dex = df.columns.get_loc('wind2bkrate')-3

        y_hat_n,ll_n = evaluate_numeric(numeric_model,X_train,X_test,y_train,y_test,num_dex)
        y_hat_c, ll_c = evaluate_categorical(cat_model,X_train,X_test,y_train,y_test,num_dex)
        y_hat_f, ll_f = evaluate_full(full_model,X_train,X_test,y_train,y_test,num_dex)
        plot_roc(y_test,[y_hat_n,y_hat_c,y_hat_f],algo+str(season),algo+str(season)+".jpg")
        write_text(ll_n, ll_c, ll_f, "../Data/TestSetResults/"+algo+str(season)+".txt",season,algo)
