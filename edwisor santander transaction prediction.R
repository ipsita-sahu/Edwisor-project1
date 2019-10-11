rm(list=ls(all=T))
setwd("G:/edwisor_project1")
getwd()

# Importing the dataset
training_dataset=read.csv("train_data.csv",header = T,na.strings = c(" ","",NA))
test_dataset=read.csv("test.csv",header = T,na.strings = c(" ","",NA))
str(training_dataset)
#check the dimension of the dataset
dim(training_dataset)
dim(test_dataset)

training_dataset=training_dataset[,2:202]
View(training_dataset)

test_dataset=test_dataset[,2:201]
View(test_dataset)
#factorize the target variable
training_dataset$target = factor(training_dataset$target, levels = c(0, 1))

#missing value analysis
sum(is.na(training_dataset))
sum(is.na(test_dataset))

numeric_index = sapply(training_dataset,is.numeric) 

numeric_data = training_dataset[,numeric_index]

cnames = colnames(numeric_data)
cnames
#boxplot diagram for training_dataset
library("ggplot2")
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "target"), data = subset(training_dataset))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="target")+
           ggtitle(paste("Box plot of target for",cnames[i])))
}
gridExtra::grid.arrange(gn1,gn5,gn2,ncol=3)
gridExtra::grid.arrange(gn6,gn7,ncol=2)
gridExtra::grid.arrange(gn8,gn9,ncol=2)

#replacing outlier with NAs
for(i in cnames){
  val = training_dataset[,i][training_dataset[,i] %in% boxplot.stats(training_dataset[,i])$out]
  print(length(val))
  training_dataset[,i][training_dataset[,i] %in% val] = NA
}

sum(is.na(training_dataset))
for(i in cnames){
  training_dataset[,i][is.na(training_dataset[,i])] = mean(training_dataset[,i], na.rm = T)
}
sum(is.na(training_dataset))


View(training_dataset)
set.seed(7)
install.packages("mlbench")
library(mlbench)
library(lattice)
library(ggplot2)
library(caret)
correlationMatrix <-cor(training_dataset[,1:200])
print(correlationMatrix)
highlyCorrelated <-findCorrelation(correlationMatrix,cutoff = 0.75)
print(highlyCorrelated)




#correlation_plot

library(corrgram)
corrgram(training_dataset[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#normalization of training_dataset
qqnorm(training_dataset$var_1)
hist(training_dataset$var_1)
for(i in cnames){
  print(i)
  training_dataset[,i] = (training_dataset[,i] - min(training_dataset[,i]))/
    (max(training_dataset[,i] - min(training_dataset[,i])))
}
 #standardization of training_dataset
  for(i in cnames){
    print(i)
    training_dataset[,i] = (training_dataset[,i] - mean(training_dataset[,i]))/
      sd(training_dataset[,i])
  }
View(training_dataset)
#normalization of test_dataset
qqnorm(test_dataset$var_1)
hist(test_dataset$var_1)
numeric_index = sapply(test_dataset,is.numeric) 

numeric_data = test_dataset[,numeric_index]

cnames_test = colnames(numeric_data)
cnames_test

for(i in cnames_test){
      print(i)
     test_dataset[,i] = (test_dataset[,i] - min(test_dataset[,i]))/
     (max(test_dataset[,i] - min(test_dataset[,i])))
}
#standardization of test_dataset
for(i in cnames_test){
  print(i)
  test_dataset[,i] = (test_dataset[,i] - mean(test_dataset[,i]))/
    sd(test_dataset[,i])
}
View(test_dataset)

library(caTools)
library(lattice)
library(ggplot2)
library(caret)
set.seed(123)
split = sample.split(training_dataset$target, SplitRatio = 0.8)
training_set = subset(training_dataset, split == TRUE)
test_set = subset(training_dataset, split == FALSE)



# Fitting Logistic Regression to the Training set


classifier = glm(formula = target ~ .,
                 family = binomial,
                 data = training_set)
summary(classifier)
# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-201])
y_pred = ifelse(prob_pred > 0.5, 1, 0)
y_pred
ConfMatrix_RF = table(test_set$target, y_pred)

confusionMatrix(ConfMatrix_RF)
(35468+1088)/(35468+512+2932+1088)

#accuracy 91.39
2932/(2932+1088)
#error_rate 72.93



#NaiveBays
library(e1071)

#Develop model
NB_model = naiveBayes(target ~ ., data = training_set)
summary(NB_model)
#predict on test cases #raw
NB_Predictions = predict(NB_model, test_set[,1:200], type = 'class')
NB_Predictions
#Look at confusion matrix
Conf_matrix = table(test_set[,201],NB_Predictions)
confusionMatrix(Conf_matrix)

#accuarcy 92.1
2581/(2581+1439)
#error_rate 64.2

#Develop Model on training data
library(C50)
C50_model =C5.0(target ~., training_set, trials = 100, rules = TRUE)

#Summary of DT model
summary(C50_model)

#write rules into disk
#write(capture.output(summary(C50_model)), "c50Rules.txt")

#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-17], type = "class")

##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$responded, C50_Predictions)
confusionMatrix(ConfMatrix_C50)

install.packages("randomForest")
library(randomForest)
library(ggplot2)
library(inTrees)
RF_model = randomForest(target ~ ., training_set, importance = TRUE, ntree = 100)
treeList = RF2List(RF_model)  
exec = extractRules(treeList, training_set[,-201])
exec[1:2,]
readableRules = presentRules(exec, colnames(training_set))
readableRules[1:2,]   
ruleMetric = getRuleMetric(exec, training_set[,-201], training_set$target)
ruleMetric[1:2,]
RF_Predictions = predict(RF_model, test_set[,-201])
ConfMatrix_RF = table(test_set$target, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

4276/(4276+0)

#Predict the test_dataset outcome
NB_Predictions = predict(NB_model, test_dataset[,1:200], type = 'class')
NB_Predictions





