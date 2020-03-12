#--------------------------------------------------------------------------------#
#----- Hand written Digit Recognition using SVM both Linear and Non-linear ------#
# Submitted by : KARPAGAM R
#--------------------------------------------------------------------------------#

# --------------------- Installing necessary packages ---------------------------#

install.packages("caret",dependencies = TRUE)
install.packages("kernlab")
install.packages("dplyr")
install.packages("readr")
install.packages("ggplot2")
install.packages("gridExtra")
install.packages("e1071")
install.packages("caTools")

#-----------------------Extracting required libraries----------------------------#

library("caret")
library("kernlab")
library("dplyr")
library("readr")
library("ggplot2")
library("gridExtra")
library("e1071")
library("caTools")


setwd("C:\\Users\\Student\\Documents\\svm")

#-----------------------Reading the given dataframes--------------------------------#

train <- read.delim("mnist_train.csv",sep=",",stringsAsFactors = FALSE,header = FALSE)
test <- read.delim("mnist_test.csv",sep=",",stringsAsFactors = FALSE,header = FALSE)


#----------Renaming the Target variable both in test and train dataframes-----------#
colnames(train)[1] <- "img_id"
colnames(test)[1] <- "img_id"

#----------------Adding train/test column in both the dataframes--------------------#

train$type <- "train"
test$type <- "test"

#--------------------Merging both the dataframes for data cleanup-------------------#

mnist_full_dataset <- rbind(train,test)

#------------------------------Checking for NAs-------------------------------------#

which(sapply(mnist_full_dataset,function(x) sum(is.na(x))) != 0) # No NAs


#--------------------Checking for redundancy among column values--------------------#

identical_cols <- which(sapply(mnist_full_dataset,function(x) length(unique(x)) == 1))
length(identical_cols) # Totally 65 identical columns present
identical_cols

mnist_clean_dataset <- mnist_full_dataset %>% select(-identical_cols)


#------------- checking for outliers at both the ends i.e. lower and uper------------#

which((mnist_clean_dataset %>% select(img_id)) > 9 | (mnist_clean_dataset %>% select(img_id)) < 0)


#-----Separating the Train and Test data for model development after data cleanup-----#

train_data <- mnist_clean_dataset %>% filter(type == "train") %>% select(-type)
test_data <- mnist_clean_dataset %>% filter(type == "test") %>% select(-type)

#------- Converting the target variable into a factor variable in both Train and Test dataset-------#

train_data$img_id <- as.factor(train_data$img_id)
test_data$img_id <- as.factor(test_data$img_id)

#----------------------Taking data for model building-------------------------------#

set.seed(100)

train_indices <- sample.split(train_data$img_id,SplitRatio = 0.3333)
train_sample <- train_data[train_indices,]

train_data_final <- train_sample
test_data_final <- test_data


#-----------------------------------Model Building--------------------------------#
#-------------------------------------LINEAR SVM----------------------------------#

linear_svm <- ksvm(img_id~., data=train_data_final, scale=FALSE, kernel="vanilladot")

linear_svm


#----------------------------Evaluating the Linear Model-------------------------#

linear_svm_evaluation <- predict(linear_svm,train_data_final)
confusionMatrix(linear_svm_evaluation,train_data_final$img_id)


#----------------Evaluating the Linear Model with test data----------------------#

linear_svm_test_evaluation <- predict(linear_svm,test_data_final)
confusionMatrix(linear_svm_test_evaluation,test_data_final$img_id)


#------------------------NON-LINEAR SVM with RBF - Kernel------------------------#

non_linear_svm_model <- ksvm(img_id~., data=train_data_final, scale=FALSE, kernel="rbfdot")

non_linear_svm_model


#----------------------------Trainig Accuracy------------------------------------#

non_linear_svm_train_evaluation <- predict(non_linear_svm_model,train_data_final)
confusionMatrix(non_linear_svm_train_evaluation,train_data_final$img_id)

#-------------------------Train Set Net Accuracy = 0.9838------------------------#

#----------------------------Test accuracy---------------------------------------#

non_linear_svm_test_evaluation <- predict(non_linear_svm_model,test_data_final)
confusionMatrix(non_linear_svm_test_evaluation,test_data_final$img_id)

#----------------------Test Net Accuracy of model = 0.9673----------------------#


#--------------------- Model Evaluation - Cross Validation----------------------#

trainControl <- trainControl(method = "cv", number = 2,verboseIter=TRUE)
metric <- "Accuracy"
set.seed(100)
grid <- expand.grid(.sigma = c(0.63e-7,1.63e-7,2.63e-7),.C=c(1,2,3))


non_linear_svm_fit <- train(img_id~.,data=train_data_final,method="svmRadial",
                                metric=metric,tuneGrid=grid,
                                trControl=trainControl)
non_linear_svm_fit

#  sigma     C  Accuracy   Kappa    
# 6.30e-08  1  0.9377406  0.9308006
# 6.30e-08  2  0.9442916  0.9380814
# 6.30e-08  3  0.9490423  0.9433616
# 1.63e-07  1  0.9567434  0.9519216
# 1.63e-07  2  0.9623443  0.9581468
# 1.63e-07  3  0.9644947  0.9605367
# 2.63e-07  1  0.9638945  0.9598701
# 2.63e-07  2  0.9684453  0.9649281
# 2.63e-07  3  0.9698455  0.9664845

#Accuracy was used to select the optimal model using  the largest value.
#The final values used for the model were sigma = 2.63e-07 and C = 3.

plot(non_linear_svm_fit)


#------------------Building a model with C = 3 and sigma = 2.63e-07----------#

non_linear_svm_final <- ksvm(img_id~.,data=train_data_final,kernel="rbfdot",
                                       scale=FALSE,C=3,kpar=list(sigma=2.63e-7))
non_linear_svm_final


#----------------------------Training accuracy-------------------------------#

non_linear_svm_train_eval1 <- predict(non_linear_svm_final,train_data_final)
confusionMatrix(non_linear_svm_train_eval1,train_data_final$img_id)

#--------------------------Net Train Accuracy = 0.9989-----------------------#



#----------------------------Test accuracy-----------------------------------#

non_linear_svm_test_evaluation1 <- predict(non_linear_svm_final,test_data_final)
confusionMatrix(non_linear_svm_test_evaluation1,test_data_final$img_id)


# Overall Statistics
# Accuracy : 0.9775
# 95% CI : (0.9744,0.9803)
# No information Rate : 0.1135
# P-Value[Acc>NIR] : <2.2e-16

#Kappa : 0.975
# Mcnemar's Test P-value : NA

