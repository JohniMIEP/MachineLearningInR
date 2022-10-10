setwd("H:/OneDrive/Studium/4. Semester/ML/Praxis")

library(e1071)
library(ANN2)
library(mboost)
library(glmnet)
library(tree)


Daten <- read.csv2("train.csv",header=TRUE,sep=",",fill=TRUE,stringsAsFactors=TRUE)

Daten$X <- NULL
Daten$id <- NULL


Daten$satisfaction <- ifelse(Daten$satisfaction == "satisfied", 1,0)
Daten[,"satisfaction"] <- as.factor(Daten[,"satisfaction"])
Daten[,"Arrival.Delay.in.Minutes"] <- as.numeric(Daten[,"Arrival.Delay.in.Minutes"])

###############################################################################
# Dimensionsreduzierung mit LASSO
###############################################################################

n <- length(Daten[,1])
Index <- sample(seq(1,n,1), replace=FALSE)
Daten <- Daten[Index,]
rownames(Daten) <- 1:n



X <- model.matrix(satisfaction ~. , Daten)
X <- X[,-1]   # entferne den Intercept
summary(X)



y <- Daten[,"satisfaction"]

model.lasso <- cv.glmnet(X,y,family = "binomial")
coef(model.lasso,s="lambda.min")

model.lasso <- cv.glmnet(X,y,family = "binomial")
coef(model.lasso,s="lambda.1se")


m <- length(X[1,])
total.numbers <- rep(0,m)

RUNS <- 100

for( run in 1:RUNS ){
  print(run)
  model.lasso <- cv.glmnet(X,y,family = "binomial")
  beta <- coef(model.lasso,s="lambda.1se")[-1,1]
  total.numbers <- total.numbers + ifelse( beta != 0, 1, 0)
  
}

total.numbers <- as.matrix(total.numbers)  # wie oft wurde welche Variable gewählt
rownames(total.numbers) <- names(beta)     # die Zeilen sollen die Namen der Variablen haben
total.numbers

########################################################################################


#Auswahl ist: Customer.Type, Age, Type.of.TravelPersonal, Class, Inflight.wifi.service, Departure.Arrival.time.convenient, Ease.of.Online.booking
#Online.boarding, Seat.comfort, Inflight.entertainment, On.board.service, Leg.room.service, Baggage.handling, Checkin.service, Inflight.service
#Cleanliness, Departure.Delay.in.Minutes, Arrival.Delay.in.Minutes

Daten.neu <- Daten[,c("Customer.Type" ,"Age", "Type.of.Travel", "Class", "Inflight.wifi.service", 
                            "Departure.Arrival.time.convenient", "Ease.of.Online.booking", "Online.boarding", "Seat.comfort", 
                            "Inflight.entertainment", "On.board.service", "Leg.room.service", "Baggage.handling", "Checkin.service", 
                            "Inflight.service", "Cleanliness", "Departure.Delay.in.Minutes", "Arrival.Delay.in.Minutes","satisfaction")]



# Zufaellige Indexierung und split in 50/50
n <- length(Daten.neu[,1])
Index <- sample(seq(1,n,1), replace=FALSE)
Daten.neu <- Daten.neu[Index,]
rownames(Daten.neu) <- 1:n

Daten.train <- Daten.neu[1:50000,]
Daten.test <- Daten.neu[50001:100000,]


##############################################################################################################
# L2 Boosting
##############################################################################################################
modelboost <- gamboost(satisfaction ~., data=Daten.train, 
                  dfbase = 4,baselearner = "bols",family = Binomial(), control = boost_control(mstop = 2000))

par(mfrow=c(1,3))
plot(modelboost)

cv10f <- cv(model.weights(modelboost), type = "kfold")
cvm <- cvrisk(modelboost, folds = cv10f, papply = lapply)
print(cvm)
mstop(cvm)
plot(cvm)

modelboost <- gamboost(satisfaction ~., 
                  data=Daten.train,baselearner = "bols",family = Binomial(), dfbase = 4, control = boost_control(mstop = mstop(cvm)))
modelboost
par(mfrow=c(1,3))
plot(modelboost)

Xboost <- Daten.test

A <- matrix(0,ncol=2,nrow=2)
colnames(A) <- c("Real: satisfied", "Real: neutral or not satisfied") 
rownames(A) <- c("Prognose: satisfied", "Prognose: neutral or not satisfied") 

prognosen.boost <- predict(modelboost,Xboost)
yboost <- Daten.test[,"satisfaction"]

A[1,1] <- sum(ifelse(yboost == 0 & prognosen.boost < 0, 1,0))
A[1,2] <- sum(ifelse(yboost == 1 & prognosen.boost < 0, 1,0))
A[2,1] <- sum(ifelse(yboost == 0 & prognosen.boost >=0, 1,0))
A[2,2] <- sum(ifelse(yboost == 1 & prognosen.boost >=0, 1,0))




####################################################################################
# Neural Network
####################################################################################
Xnn <- model.matrix(satisfaction ~., Daten.train)
Xnn <- Xnn[,-1]
summary(Xnn)

ynn <- Daten.train[,"satisfaction"]

modelnn <- neuralnetwork(Xnn, ynn, hidden.layers=c(10,8), regression = FALSE, 
                       loss.type = "log", learn.rates = 1e-04,n.epochs = 1000,
                       verbose=FALSE)

plot(modelnn)

Xnntest <- model.matrix(satisfaction ~., Daten.test)
Xnntest <- Xnntest[,-1]


predict(modelnn,Xnntest)$predictions


B <- matrix(0,ncol=2,nrow=2)
colnames(B) <- c("Real: satisfied", "Real: neutral or not satisfied") 
rownames(B) <- c("Prognose: satisfied", "Prognose: neutral or not satisfied") 

prognosen.nn <- predict(modelnn,Xtest)$predictions
ynntest <- Daten.test[,"satisfaction"]

B[1,1] <- sum(ifelse(ynntest == 0 & prognosen.nn == 0, 1,0))
B[1,2] <- sum(ifelse(ynntest == 1 & prognosen.nn == 0, 1,0))
B[2,1] <- sum(ifelse(ynntest == 0 & prognosen.nn == 1, 1,0))
B[2,2] <- sum(ifelse(ynntest == 1 & prognosen.nn == 1, 1,0))



##################################################################################
# Entscheidungsbaum
##################################################################################
Baum <- tree(satisfaction ~., data=Daten.train)
plot(Baum)
text(Baum)
tuning <- cv.tree(Baum, K=20)
tuning
plot(tuning)


t <- which.min(tuning$dev)
Anzahl.Endknoten <- tuning$size[t]

modelbaum <- prune.tree(Baum,best=Anzahl.Endknoten)
plot(modelbaum)
text(modelbaum)


Xbaum <- Daten.test[,c("Customer.Type" ,"Age", "Type.of.Travel", "Class", "Inflight.wifi.service", 
              "Departure.Arrival.time.convenient", "Ease.of.Online.booking", "Online.boarding", "Seat.comfort", 
              "Inflight.entertainment", "On.board.service", "Leg.room.service", "Baggage.handling", "Checkin.service", 
              "Inflight.service", "Cleanliness", "Departure.Delay.in.Minutes", "Arrival.Delay.in.Minutes")]


C <- matrix(0,ncol=2,nrow=2)
colnames(C) <- c("Real: satisfied", "Real: neutral or not satisfied") 
rownames(C) <- c("Prognose: satisfied", "Prognose: neutral or not satisfied") 

prognosen.tree <- predict(modelbaum,Xbaum)
prognosen.tree

prognosen.tree <- round(prognosen[,2])

ytest <- Daten.test[,"satisfaction"]

C[1,1] <- sum(ifelse(ytest == 0 & prognosen.tree == 0, 1,0))
C[1,2] <- sum(ifelse(ytest == 1 & prognosen.tree == 0, 1,0))
C[2,1] <- sum(ifelse(ytest == 0 & prognosen.tree == 1, 1,0))
C[2,2] <- sum(ifelse(ytest == 1 & prognosen.tree == 1, 1,0))



###########################################################################################
# Confusion matrix plot
###########################################################################################
atable <- as.table(matrix(c(A[1,1], A[1,2], A[2,1], A[2,2]), nrow = 2, byrow = TRUE))
fourfoldplot(atable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "L2",std = "all.max")

btable <- as.table(matrix(c(B[1,1], B[1,2], B[2,1], B[2,2]), nrow = 2, byrow = TRUE))
fourfoldplot(btable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Neural network",std = "all.max")

ctable <- as.table(matrix(c(C[1,1], C[1,2], C[2,1], C[2,2]), nrow = 2, byrow = TRUE))
fourfoldplot(ctable, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Tree",std = "all.max")

