setwd("~/Desktop/MachineLearning/Project/MLaCSProject/Code/")
x <- read.csv("../Data/MLB_AtBats_RegularSeason_2012_sorted.csv")
#install.packages("lme4")

library("lme4")
library("pROC")
make_model <- function(seas){
  print(paste("making model for ",seas,"season"))
  fname<- paste("../Data/MLB_AtBats_RegularSeason_",seas,"_sorted.csv",sep="")
  df <- read.csv(fname)
  df$K <- (df$descr=="Strikeout")
  mycols <- c('K','pitcher','batter','stadium')
  df<-df[mycols]
  trainrows = floor(nrow(df)*0.8)
  testrows = nrow(df)-trainrows
  traindf <- head(df,trainrows)
  testdf <- tail(df,testrows)
  print("fitting model")
  
  fit <- glmer(K~(1|pitcher)+(1|batter)+(1|stadium),traindf,family=binomial)
  print("model fit")
  preds<-predict(fit,newdata=testdf,type="response",allow.new.levels=TRUE)
  print("prediction")
  write.csv(preds,paste("../Data/lme_",seas,".csv",sep=""))
  print("csv written ")
  
  
}



for(i in 2012:2017){
  make_model(i)
  
}



yhat =c(.5,.2,.1,.9)
y = c(1,0,1,1)

library("pROC")

plot.roc(y,yhat)

fixef(fit)
head(ranef(fit)$batter)
head(ranef(fit)$pitcher)
head(ranef(fit)$stadium)