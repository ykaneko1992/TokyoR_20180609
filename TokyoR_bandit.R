library(dplyr)
library(glmnet)
library(ggplot2)


set.seed(2017)

tes <- rnorm(2000)
for_Xi0 <- tes[tes < -0.85 & -1.15 < tes]
tes2 <- rnorm(2000)
for_Xi1 <-tes2[tes2 < -0.85 & -1.15 < tes2]

#グラフ用データセット用意
x <- seq(-2, 2, by = 0.02)
grid <- expand.grid(x, x)
colnames(grid) <- c("X_0", "X_1")
graph.bandit <- grid %>% 
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2)
  
#tes <- rowMeans(graph.bandit)
#tes2 <- apply(graph.bandit,1,var)
#sample_n(tbl = graph.bandit, size = 10, replace = T)


#コールドスタートデータ用意
X_i0 <- sample(for_Xi0,50)
X_i1 <- sample(for_Xi1,50)
eps <- rnorm(50, mean=0, sd= 1.5)
Y_i0 <- 0.5*(X_i0 + 1)^2 + 0.5*(X_i1 + 1)^2 + eps
Y_i1 <- 1 + eps
Y_i2 <- 2 - 0.5*(X_i0 + 1)^2 - 0.5*(X_i1 + 1)^2 + eps

cold_start <- data.frame(X_0=X_i0, 
                         X_1=X_i1, 
                         Y_0=Y_i0,
                         Y_1=Y_i1,
                         Y_2=Y_i2)

cold_reward <- cold_start[3:5]

#batch1以降のデータ用意
b_X_i0 <- rnorm(220)
b_X_i1 <- rnorm(220) 
b_eps <- rnorm(220, mean=0, sd= 1.5)
b_Y_i0 <- 0.5*(b_X_i0 + 1)^2 + 0.5*(b_X_i1 + 1)^2 + b_eps
b_Y_i1 <- 1 + b_eps
b_Y_i2 <- 2 - 0.5*(b_X_i0 + 1)^2 - 0.5*(b_X_i1 + 1)^2 + b_eps

batch <- data.frame(X_0=b_X_i0, 
                    X_1=b_X_i1, 
                    Y_0=b_Y_i0,
                    Y_1=b_Y_i1,
                    Y_2=b_Y_i2)

batch_reward <- batch[3:5]


#######cold_start

#random assignment

first_run <- floor(runif(50, min=0, max=2+1))


n <- 4
result <- data.frame(matrix(rep(NA, n), nrow=1))[numeric(0), ]
colnames(result) <- c("X_0", "X_1", "arm","reward")

for (i in c(1:50)) {
  X_0 <- cold_start[i,1]
  X_1 <- cold_start[i,2]
  arm <- first_run[i]
  reward <- cold_reward[i,arm+1]
  result[i,] <- c( X_0,X_1,arm,reward)
}

#リッジ回帰
arm_0 <- result %>% 
  filter(arm == 0) 

arm0.sampled <- arm_0 %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2) %>%
  mutate(rew = reward) %>%
  select(-arm, -reward)

arm_1 <- result %>% 
  filter(arm == 1)

arm1.sampled <- arm_1 %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2) %>%
  mutate(rew = reward) %>%
  select(-arm, -reward)

arm_2 <- result %>% 
  filter(arm == 2)

arm2.sampled <- arm_2 %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2) %>%
  mutate(rew = reward) %>%
  select(-arm, -reward)

######bootstrap glmnet

batch_pred <- batch[c(1:10),c(1:2)] %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2)

#arm0
mean_arm_0 <- NULL
pred.arm0.graph <- NULL

for (i in c(1:100)) {
  arm0.glm <- sample_n(tbl = arm0.sampled, size = nrow(arm0.sampled), replace = T)
  arm_0_glm<- cv.glmnet(as.matrix(arm0.glm[1:4]),as.matrix(arm0.glm[5]),family="gaussian",alpha=0,standardize = F)
  mean_arm_0 <- cbind(mean_arm_0,predict(arm_0_glm, newx=as.matrix(batch_pred)))
  pred.arm0.graph <- cbind(pred.arm0.graph,predict(arm_0_glm, newx=as.matrix(graph.bandit)))
}

arm0.glm[1:4]
pred_arm_0 <- apply(mean_arm_0,1,mean)
var.arm0 <- apply(mean_arm_0,1,var)
pred.graph.arm0 <- apply(pred.arm0.graph,1,mean)
var.graph.arm0 <- apply(pred.arm0.graph,1,var)

#arm1
mean_arm_1 <- NULL
pred.arm1.graph <- NULL

for (i in c(1:100)) {
  arm1.glm <- sample_n(tbl = arm1.sampled, size = nrow(arm1.sampled), replace = T)
  arm_1_glm<- cv.glmnet(as.matrix(arm1.glm[1:4]),as.matrix(arm1.glm[5]),family="gaussian",alpha=0,standardize = F)
  mean_arm_1 <- cbind(mean_arm_1,predict(arm_1_glm, newx=as.matrix(batch_pred)))
  pred.arm1.graph <- cbind(pred.arm1.graph,predict(arm_1_glm, newx=as.matrix(graph.bandit)))
}

pred_arm_1 <- apply(mean_arm_1,1,mean)
var.arm1 <- apply(mean_arm_1,1,var)
pred.graph.arm1 <- apply(pred.arm1.graph,1,mean)
var.graph.arm1 <- apply(pred.arm1.graph,1,var)

#arm2
mean_arm_2 <- NULL
pred.arm2.graph <- NULL

for (i in c(1:100)) {
  arm2.glm <- sample_n(tbl = arm2.sampled, size = nrow(arm2.sampled), replace = T)
  arm_2_glm<- cv.glmnet(as.matrix(arm2.glm[1:4]),as.matrix(arm2.glm[5]),family="gaussian",alpha=0,standardize = F)
  mean_arm_2 <- cbind(mean_arm_2,predict(arm_2_glm, newx=as.matrix(batch_pred)))
  pred.arm2.graph <- cbind(pred.arm2.graph,predict(arm_2_glm, newx=as.matrix(graph.bandit)))
}

pred_arm_2 <- apply(mean_arm_2,1,mean)
var.arm2 <- apply(mean_arm_2,1,var)
pred.graph.arm2 <- apply(pred.arm2.graph,1,mean)
var.graph.arm2 <- apply(pred.arm2.graph,1,var)


#######batch part

arm_0.TS.glm <- arm0.sampled
arm_1.TS.glm <- arm1.sampled
arm_2.TS.glm <- arm2.sampled
arm_0.UCB.glm <- arm0.sampled
arm_1.UCB.glm <- arm1.sampled
arm_2.UCB.glm <- arm2.sampled

pred_arm_0.TS <- pred_arm_0
pred_arm_1.TS <- pred_arm_1
pred_arm_2.TS <- pred_arm_2
pred_arm_0.UCB <- pred_arm_0
pred_arm_1.UCB <- pred_arm_1
pred_arm_2.UCB <- pred_arm_2

var.arm0.TS <- var.arm0
var.arm1.TS <- var.arm1
var.arm2.TS <- var.arm2
var.arm0.UCB <- var.arm0
var.arm1.UCB <- var.arm1
var.arm2.UCB <- var.arm2

pred.graph.arm0.TS <- pred.arm0.graph
pred.graph.arm1.TS <- pred.arm1.graph
pred.graph.arm2.TS <- pred.arm2.graph
pred.graph.arm0.UCB <- pred.arm0.graph
pred.graph.arm1.UCB <- pred.arm1.graph
pred.graph.arm2.UCB <- pred.arm2.graph

var.graph.arm0.TS <- var.graph.arm0
var.graph.arm1.TS <- var.graph.arm1
var.graph.arm2.TS <- var.graph.arm2
var.graph.arm0.UCB <- var.graph.arm0
var.graph.arm1.UCB <- var.graph.arm1
var.graph.arm2.UCB <- var.graph.arm2

###TS assignment

#batch assignmen
for (k in c(1:21)) {
  
arm.TS = c(1:10)

for (i in c(1:10)) {
   assign.arm0.TS <- rnorm(1, mean=pred_arm_0.TS[i], sd=var.arm0.TS[i])
   assign.arm1.TS <- rnorm(1, mean=pred_arm_1.TS[i], sd=var.arm1.TS[i])
   assign.arm2.TS <- rnorm(1, mean=pred_arm_2.TS[i], sd=var.arm2.TS[i])
   max.assign <- max(assign.arm0.TS,assign.arm1.TS,assign.arm2.TS)
   if (max.assign == assign.arm0.TS) {
     arm.TS[i] = 0
   } else if (max.assign == assign.arm1.TS) {
     arm.TS[i] = 1
   } else arm.TS[i] = 2
}

#graph assignment
arm.TS.graph = c(1:length(grid[,1]))
len <- length(grid[,1])

for (i in c(1:len)) {
  assign.arm0.TS <- rnorm(1, mean=pred.graph.arm0.TS[i], sd=var.graph.arm0.TS[i])
  assign.arm1.TS <- rnorm(1, mean=pred.graph.arm1.TS[i], sd=var.graph.arm1.TS[i])
  assign.arm2.TS <- rnorm(1, mean=pred.graph.arm2.TS[i], sd=var.graph.arm2.TS[i])
  max.assign <- max(assign.arm0.TS,assign.arm1.TS,assign.arm2.TS)
  if (max.assign == assign.arm0.TS) {
    arm.TS.graph[i] = 0
  } else if (max.assign == assign.arm1.TS) {
    arm.TS.graph[i] = 1
  } else arm.TS.graph[i] = 2
}


###UCB assignment

#batch assignment
arm.UCB = c(1:10)

for (i in c(1:10)) {
  assign.arm0.UCB <- pred_arm_0[i] + sqrt(2*log(i))*var.arm0.UCB[i]
  assign.arm1.UCB <- pred_arm_1[i] + sqrt(2*log(i))*var.arm1.UCB[i]
  assign.arm2.UCB <- pred_arm_2[i] + sqrt(2*log(i))*var.arm2.UCB[i]
  max.assign <- max(assign.arm0.UCB,assign.arm1.UCB,assign.arm2.UCB)
  if (max.assign == assign.arm0.UCB) {
    arm.UCB[i] = 0
  } else if (max.assign == assign.arm1.TS) {
    arm.UCB[i] = 1
  } else arm.UCB[i] = 2
}

#graph assignment
arm.UCB.graph = c(1:length(grid[,1]))
len <- length(grid[,1])

for (i in c(1:len)) {
  assign.arm0.UCB <- pred.graph.arm0.UCB[i] + sqrt(2*log(i))*var.graph.arm0.UCB[i]
  assign.arm1.UCB <- pred.graph.arm1.UCB[i] + sqrt(2*log(i))*var.graph.arm1.UCB[i]
  assign.arm2.UCB <- pred.graph.arm2.UCB[i] + sqrt(2*log(i))*var.graph.arm2.UCB[i]
  max.assign <- max(assign.arm0.UCB,assign.arm1.UCB,assign.arm2.UCB)
  if (max.assign == assign.arm0.UCB) {
    arm.UCB.graph[i] = 0
  } else if (max.assign == assign.arm1.UCB) {
    arm.UCB.graph[i] = 1
  } else arm.UCB.graph[i] = 2
}

###graph
graph.TS <- grid %>%
  mutate(arm = arm.TS.graph)

TS_batch = paste("TS Batch " , as.character(k-1))
TS <- ggplot(graph.TS, aes(y = X_0, x = X_1, colour = factor(arm))) + geom_point() + labs(title= TS_batch)
file.name.TS = paste("TS_Batch_", as.character(k-1),".png", collapse="")
print(file.name.TS)
ggsave(file = file.name.TS, plot = TS)

graph.UCB <- grid %>%
  mutate(arm = arm.UCB.graph)

UCB_batch = paste("UCB Batch " , as.character(k-1))
UCB <- ggplot(graph.UCB, aes(y = X_0, x = X_1, colour = factor(arm))) + geom_point() + labs(title= UCB_batch)
file.name.UCB <- paste("UCB_Batch_", as.character(k-1),".png", collapse="")
print(file.name.UCB)
ggsave(file = file.name.UCB, plot = UCB)

####for next batch
#get TS rewards
n <- 4
reward.TS <- data.frame(matrix(rep(NA, n), nrow=1))[numeric(0), ]
colnames(reward.TS) <- c("X_0", "X_1", "arm","reward")

for (i in c(1:10)) {
  X_0 <- batch[k*10+i,1]
  X_1 <- batch[k*10+i,2]
  arm <- arm.TS[i]
  reward <- batch_reward[k*10+i,arm+1]
  reward.TS[i,] <- c( X_0,X_1,arm,reward)
}

#get UCB rewards
n <- 4
reward.UCB <- data.frame(matrix(rep(NA, n), nrow=1))[numeric(0), ]
colnames(reward.UCB) <- c("X_0", "X_1", "arm","reward")

for (i in c(1:10)) {
  X_0 <- batch[k*10+i,1]
  X_1 <- batch[k*10+i,2]
  arm <- arm.UCB[i]
  reward <- batch_reward[k*10+i,arm+1]
  reward.UCB[i,] <- c( X_0,X_1,arm,reward)
}

###Ridge TS 

arm_0.TS <- reward.TS %>% 
  filter(arm == 0) 

arm0_2.TS <- arm_0.TS %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2) %>%
  mutate(rew = reward) %>%
  select(-arm,-reward)

arm_0.TS.glm <- rbind(arm_0.TS.glm , arm0_2.TS)


arm_1.TS <- reward.TS %>% 
  filter(arm == 1) 

arm1_2.TS <- arm_1.TS %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2) %>%
  mutate(rew = reward) %>%
  select(-arm,-reward)

arm_1.TS.glm <- rbind(arm_1.TS.glm , arm1_2.TS)


arm_2.TS <- reward.TS %>% 
  filter(arm == 2) 

arm2_2.TS <- arm_2.TS %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2) %>%
  mutate(rew = reward) %>%
  select(-arm,-reward)

arm_2.TS.glm <- rbind(arm_2.TS.glm , arm2_2.TS)


batch_pred <- batch[seq(k*10+1,(k+1)*10, by =1),c(1:2)] %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2)

#arm0
mean_arm_0 <- NULL
pred.arm0.graph <- NULL

for (i in c(1:100)) {
  arm0.glm <- sample_n(tbl = arm_0.TS.glm, size = nrow(arm_0.TS.glm), replace = T)
  arm_0_glm<- cv.glmnet(as.matrix(arm0.glm[1:4]),as.matrix(arm0.glm[5]),family="gaussian",alpha=0,standardize = F)
  mean_arm_0 <- cbind(mean_arm_0,predict(arm_0_glm, newx=as.matrix(batch_pred)))
  pred.arm0.graph <- cbind(pred.arm0.graph,predict(arm_0_glm, newx=as.matrix(graph.bandit)))
}

pred_arm_0.TS <- apply(mean_arm_0,1,mean)
var.arm0.TS <- apply(mean_arm_0,1,var)
pred.graph.arm0.TS <- apply(pred.arm0.graph,1,mean)
var.graph.arm0.TS <- apply(pred.arm0.graph,1,var)

#arm1 
mean_arm_1 <- NULL
pred.arm1.graph <- NULL

for (i in c(1:100)) {
  arm1.glm <- sample_n(tbl = arm_1.TS.glm, size = nrow(arm_1.TS.glm), replace = T)
  arm_1_glm<- cv.glmnet(as.matrix(arm1.glm[1:4]),as.matrix(arm1.glm[5]),family="gaussian",alpha=0,standardize = F)
  mean_arm_1 <- cbind(mean_arm_1,predict(arm_1_glm, newx=as.matrix(batch_pred)))
  pred.arm1.graph <- cbind(pred.arm1.graph,predict(arm_1_glm, newx=as.matrix(graph.bandit)))
}

pred_arm_1.TS <- apply(mean_arm_1,1,mean)
var.arm1.TS <- apply(mean_arm_1,1,var)
pred.graph.arm1.TS <- apply(pred.arm1.graph,1,mean)
var.graph.arm1.TS <- apply(pred.arm1.graph,1,var)

#arm2
mean_arm_2 <- NULL
pred.arm2.graph <- NULL

for (i in c(1:100)) {
  arm2.glm <- sample_n(tbl = arm_2.TS.glm, size = nrow(arm_2.TS.glm), replace = T)
  arm_2_glm<- cv.glmnet(as.matrix(arm2.glm[1:4]),as.matrix(arm2.glm[5]),family="gaussian",alpha=0,standardize = F)
  mean_arm_2 <- cbind(mean_arm_2,predict(arm_2_glm, newx=as.matrix(batch_pred)))
  pred.arm2.graph <- cbind(pred.arm2.graph,predict(arm_2_glm, newx=as.matrix(graph.bandit)))
}

pred_arm_2.TS <- apply(mean_arm_2,1,mean)
var.arm2.TS <- apply(mean_arm_2,1,var)
pred.graph.arm2.TS <- apply(pred.arm2.graph,1,mean)
var.graph.arm2.TS <- apply(pred.arm2.graph,1,var)

###Ridge UCB 

arm_0.UCB <- reward.UCB %>% 
  filter(arm == 0) 

arm0_2.UCB <- arm_0.UCB %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2) %>%
  mutate(rew = reward) %>%
  select(-arm,-reward)

arm_0.UCB.glm <- rbind(arm_0.UCB.glm , arm0_2.UCB)


arm_1.UCB <- reward.UCB %>% 
  filter(arm == 1) 

arm1_2.UCB <- arm_1.UCB %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2) %>%
  mutate(rew = reward) %>%
  select(-arm,-reward)

arm_1.UCB.glm <- rbind(arm_1.UCB.glm , arm1_2.UCB)


arm_2.UCB <- reward.UCB %>% 
  filter(arm == 2) 

arm2_2.UCB <- arm_2.UCB %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2) %>%
  mutate(rew = reward) %>%
  select(-arm,-reward)

arm_2.UCB.glm <- rbind(arm_2.UCB.glm , arm2_2.UCB)

batch_pred <- batch[seq(k*10+1,(k+1)*10, by =1),c(1:2)] %>%
  mutate(X_0_quad = X_0^2) %>%
  mutate(X_1_quad = X_1^2)

#arm0
mean_arm_0 <- NULL
pred.arm0.graph <- NULL

for (i in c(1:100)) {
  arm0.glm <- sample_n(tbl = arm_0.UCB.glm, size = nrow(arm_0.UCB.glm), replace = T)
  arm_0_glm<- cv.glmnet(as.matrix(arm0.glm[1:4]),as.matrix(arm0.glm[5]),family="gaussian",alpha=0,standardize = F)
  mean_arm_0 <- cbind(mean_arm_0,predict(arm_0_glm, newx=as.matrix(batch_pred)))
  pred.arm0.graph <- cbind(pred.arm0.graph,predict(arm_0_glm, newx=as.matrix(graph.bandit)))
}

pred_arm_0.UCB <- apply(mean_arm_0,1,mean)
var.arm0.UCB <- apply(mean_arm_0,1,var)
pred.graph.arm0.UCB <- apply(pred.arm0.graph,1,mean)
var.graph.arm0.UCB <- apply(pred.arm0.graph,1,var)

#arm1 
mean_arm_1 <- NULL
pred.arm1.graph <- NULL

for (i in c(1:100)) {
  arm1.glm <- sample_n(tbl = arm_1.UCB.glm, size = nrow(arm_1.UCB.glm), replace = T)
  arm_1_glm<- cv.glmnet(as.matrix(arm1.glm[1:4]),as.matrix(arm1.glm[5]),family="gaussian",alpha=0,standardize = F)
  mean_arm_1 <- cbind(mean_arm_1,predict(arm_1_glm, newx=as.matrix(batch_pred)))
  pred.arm1.graph <- cbind(pred.arm1.graph,predict(arm_1_glm, newx=as.matrix(graph.bandit)))
}

pred_arm_1.UCB <- apply(mean_arm_1,1,mean)
var.arm1.UCB <- apply(mean_arm_1,1,var)
pred.graph.arm1.UCB <- apply(pred.arm1.graph,1,mean)
var.graph.arm1.UCB <- apply(pred.arm1.graph,1,var)

#arm2
mean_arm_2 <- NULL
pred.arm2.graph <- NULL

for (i in c(1:100)) {
  arm2.glm <- sample_n(tbl = arm_2.UCB.glm, size = nrow(arm_2.UCB.glm), replace = T)
  arm_2_glm<- cv.glmnet(as.matrix(arm2.glm[1:4]),as.matrix(arm2.glm[5]),family="gaussian",alpha=0,standardize = F)
  mean_arm_2 <- cbind(mean_arm_2,predict(arm_2_glm, newx=as.matrix(batch_pred)))
  pred.arm2.graph <- cbind(pred.arm2.graph,predict(arm_2_glm, newx=as.matrix(graph.bandit)))
}

pred_arm_2.UCB <- apply(mean_arm_2,1,mean)
var.arm2.UCB <- apply(mean_arm_2,1,var)
pred.graph.arm2.UCB <- apply(pred.arm2.graph,1,mean)
var.graph.arm2.UCB <- apply(pred.arm2.graph,1,var)
}