import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import GridSearchCV, PredefinedSplit,ParameterGrid
from sklearn.metrics import mean_squared_error, make_scorer,log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from tt_split import *
import time
import scipy.special
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, roc_curve


FIG_PATH="../Figs/"
DATA_PATH="../Data/"
mypath = "Models"



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

def cross_validate(X_train, X_val,y_train, y_val,param_grid):
	print("cross validating model.")
	X_train_val = np.vstack((X_train, X_val))
	y_train_val = np.concatenate((y_train, y_val))
	val_fold = [-1]*len(X_train) + [0]*len(X_val) # 0 corresponds to validation
	estimator_ = LogisticRegression()
	grid = GridSearchCV(estimator_,param_grid,return_train_score=True,
		cv = PredefinedSplit(test_fold=val_fold),refit = True,scoring = 'neg_log_loss',verbose=3)
	grid.fit(X_train_val, y_train_val)
	return grid.best_estimator_, grid.cv_results_['params'][grid.best_index_]

def evaluate_numeric(model,X_train, X_test, y_train,y_test,numdex,name,scale=True):
	X_train = X_train[:,0:numdex]
	X_test = X_test[:,0:numdex]

	if scale:
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train=scaler.transform(X_train)
		X_test=scaler.transform(X_test)
	model.fit(X_train,y_train)
	y_hat = model.predict_proba(X_test)
	joblib.dump(model, name)
	ll = log_loss(y_test,y_hat)
	return y_hat,ll

def evaluate_categorical(model, X_train, X_test, y_train, y_test, catdex,name):
	X_train =X_train[:,catdex:]
	X_test = X_test[:,catdex:]
	model.fit(X_train,y_train)
	y_hat = model.predict_proba(X_test)
	ll = log_loss(y_test,y_hat)
	joblib.dump(model,name)
	return y_hat,ll

def evaluate_full(model, X_train, X_test, y_train, y_test, splitdex,name,scale=True):

	if scale:
		scaler = StandardScaler()
		scaler.fit(X_train[:,0:splitdex])
		X_train[:,0:splitdex]=scaler.transform(X_train[:,0:splitdex])
		X_test[:,0:splitdex]=scaler.transform(X_test[:,0:splitdex])
	model.fit(X_train,y_train)
	joblib.dump(model,name)
	y_hat = model.predict_proba(X_test)
	ll = log_loss(y_test,y_hat)
	return y_hat,ll

def evaluate_full1(model, X_train, X_test, y_train, y_test, splitdex,scale=True):

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

def make_categorical_only(X_train,X_val, y_train,y_val, catdex):
	#pgrid={'hidden_layer_sizes':[(10,),(100,),(100,100),(10,20),(30,30,30,30),(100,100,100),(1000,)], 'activation':['tanh','relu'],'alpha':np.logspace(-5,-1,10)}
	print("making categorical only")
	pgrid={'penalty':['l1','l2'],'C':[1,10,100],'fit_intercept':[True,False]}
	X_train =X_train[:,catdex:]
	X_val = X_val[:,catdex:]
	best_cat_model, ParamDict= cross_validate(X_train, X_val, y_train, y_val,pgrid)
	return best_cat_model

def make_numeric_only(X_train,X_val, y_train, y_val, numdex, scale=True):
	#pgrid={'hidden_layer_sizes':[(10,),(100,),(100,100),(10,20),(30,30,30,30),(100,100,100),(1000,)], 'activation':['tanh','relu'],'alpha':np.logspace(-5,-1,10)}

	print("making numeric only")
	pgrid={'penalty':['l1','l2'],'C':[1,10,100],'fit_intercept':[True,False]}

	X_train = X_train[:,0:numdex]
	X_val = X_val[:,0:numdex]
	if scale:
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train=scaler.transform(X_train)
		X_val=scaler.transform(X_val)
	best_numeric_model, ParamDict= cross_validate(X_train, X_val, y_train, y_val,pgrid)
	return best_numeric_model

def make_full_model(X_train,X_val,y_train,y_val,splitdex,scale=True):
	print("making full model")
	#pgrid={'hidden_layer_sizes':[(10,),(100,),(100,100),(10,20),(30,30,30,30),(100,100,100),(1000,)], 'activation':['tanh','relu'],'alpha':np.logspace(-5,-1,10)}
	pgrid={'penalty':['l1','l2'],'C':[1,10,100],'fit_intercept':[True,False]}



	if scale:
		scaler = StandardScaler()
		scaler.fit(X_train[:,0:splitdex])
		X_train[:,0:splitdex]=scaler.transform(X_train[:,0:splitdex])
		X_val[:,0:splitdex]=scaler.transform(X_val[:,0:splitdex])
	best_full_model, ParamDict= cross_validate(X_train, X_val, y_train, y_val,pgrid)
	print(best_full_model)
	return best_full_model


def write_text(numeric,cat,full,fname,season,model_type="Adaboost Classifier"):
	ostream = open(fname, "w")
	ostream.write("Results for "+str(model_type)+" in "+str(season)+"\n")
	ostream.write("\t numeric log loss: "+str(numeric)+"\n")
	ostream.write("\t categorical log loss: "+str(cat)+"\n")
	ostream.write("\t full model loss: "+str(full)+"\n")
	ostream.close()

def ifoneorzero(x):
	if x == 1:
		x = 0.99999
	elif x == 0:
		x = 0.00001
	return x
def main():

    Seasons=["2012"]#,"2013","2014","2015","2016","2017"]
    for season in Seasons:
        print("Fitting "+str(season)+" season")
        file_name = "../Data/RegularSeasonFeatures"+str(season)+".csv"
        df = pd.read_csv(file_name,index_col=0)
        #print(df.shape)
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_season(df,validation=True)
        train_names = X_train[:,0:3]
        val_names = X_val[:,0:3]
        X_train = X_train[:,3:]
        X_val = X_val[:,3:]
	X_test = X_test[:,3:]
        num_dex = df.columns.get_loc('wind2bkrate')-3
	print(X_train.shape,X_val.shape,X_test.shape)
        X = np.vstack((X_train, X_val, X_test))
        Y = np.concatenate((y_train, y_val, y_test))
        #print(X_train.shape,X_test.shape,X.shape)

        #print(y_train.shape,y_test.shape)
        #Y = np.append(y_train,y_test)
        #print(y_train.shape,y_test.shape,Y.shape)
        #print(X.shape)
        full_model_ada = joblib.load(mypath + "/" + "Ada" + "_full_" + season + ".pkl")
        full_model_extra = joblib.load(mypath + "/" + "ExtraTree" + "_full_" + season + ".pkl")
        full_model_MLP = joblib.load(mypath + "/" + "MLP" + "_full_" + season + ".pkl")
        full_model_RanFor = joblib.load(mypath + "/" + "RanFor" + "_full_" + season + ".pkl")
        #full_model_gbm = joblib.load(mypath + "/" + "GBM" + "_full_" + season + ".pkl")
        full_model_ada.fit(X_train,y_train)
        full_model_extra.fit(X_train,y_train)
        full_model_RanFor.fit(X_train,y_train)
        full_model_MLP.fit(X_train,y_train)
        #full_model_gbm.fit(X_train, y_train)
        Ada_y_hat_f = full_model_ada.predict_proba(X)
        Extra_y_hat_f = full_model_extra.predict_proba(X)
        RanFor_y_hat_f = full_model_RanFor.predict_proba(X)
        MLP_y_hat_f = full_model_MLP.predict_proba(X)
        #GBM_y_hat_f = full_model_gbm.predict_proba(X)
        #print(MLP_y_hat_f)
        df['pitcherkrate'] = df['pitcherkrate'].apply(lambda x: scipy.special.logit(ifoneorzero(x)))
        df['batterkrate'] = df['batterkrate'].apply(lambda x: scipy.special.logit(ifoneorzero(x)))

        df['Ada'] = pd.Series(Ada_y_hat_f[:,0], index=df.index)
        df['Extra'] = pd.Series(Extra_y_hat_f[:,0], index=df.index)
        df['MLP'] = pd.Series(MLP_y_hat_f[:,0], index=df.index)
        df['RanFor'] = pd.Series(RanFor_y_hat_f[:,0], index=df.index)
        #df['GBM'] = pd.Series(GBM_y_hat_f[:,0], index=df.index)
        #print(df['MLP'].max())


	numeric_model = joblib.load(mypath + "/" + "LogReg" + "_num_" + season + ".pkl")


                #make categorical model for the season, evaluate it pickle it
        cat_model = joblib.load(mypath + "/" + "LogReg" + "_cat_" + season + ".pkl")


                #make full model for the season, evaluate it pickle it
        full_model = joblib.load(mypath + "/" + "LogReg" + "_full_" + season + ".pkl")

	X_train, X_test, y_train, y_test = train_test_split_season(df)
        train_names = X_train[:,0:3]
        test_names = X_test[:,0:3]

        X_train = X_train[:,3:]
        X_test = X_test[:,3:]
	X = np.vstack((X_train, X_test))

	full_model_ada = joblib.load(mypath + "/" + "Ada" + "_full_" + season + ".pkl")
    full_model_extra = joblib.load(mypath + "/" + "ExtraTree" + "_full_" + season + ".pkl")
    full_model_MLP = joblib.load(mypath + "/" + "MLP" + "_full_" + season + ".pkl")
    full_model_RanFor = joblib.load(mypath + "/" + "RanFor" + "_full_" + season + ".pkl")

    full_model_ada.fit(X_train,y_train)
    full_model_extra.fit(X_train,y_train)
    full_model_RanFor.fit(X_train,y_train)
    full_model_MLP.fit(X_train,y_train)
        #full_model_gbm.fit(X_train, y_train)
    Ada_y_hat_f = full_model_ada.predict_proba(X)
    Extra_y_hat_f = full_model_extra.predict_proba(X)
    RanFor_y_hat_f = full_model_RanFor.predict_proba(X)
    MLP_y_hat_f = full_model_MLP.predict_proba(X)
    #GBM_y_hat_f = full_model_gbm.predict_proba(X)
    df['Ada'] = pd.Series(Ada_y_hat_f[:,0], index=df.index)
    df['Extra'] = pd.Series(Extra_y_hat_f[:,0], index=df.index)
    df['MLP'] = pd.Series(MLP_y_hat_f[:,0], index=df.index)
    df['RanFor'] = pd.Series(RanFor_y_hat_f[:,0], index=df.index)
    #df['GBM'] = pd.Series(GBM_y_hat_f[:,0], index=df.index)

    y_hat_n,ll_n = evaluate_numeric(numeric_model,X_train,X_test,y_train,y_test,num_dex, "./Models/HybridLogReg_num_"+str(season)+".pkl")
    y_hat_c, ll_c = evaluate_categorical(cat_model,X_train,X_test,y_train,y_test,num_dex, "./Models/HybridLogReg_cat_"+str(season)+".pkl")
    y_hat_f, ll_f = evaluate_full(full_model,X_train,X_test,y_train,y_test,num_dex, "./Models/HybridLogReg_full_"+str(season)+".pkl")
    write_text(ll_n, ll_c, ll_f, "../Data/HybridLogReg_results_"+str(season)+".txt",season)
    #plot_roc(y_test,[y_hat_n,y_hat_c,y_hat_f],"Logistic Regression ROC Curves " + str(season)+ " Season","../Figs/LogReg_ROC_"+str(season)+".pdf")

if __name__ == '__main__':
	main()
