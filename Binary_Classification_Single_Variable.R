#########################################################################################################
# Initial Setup of working directory and functions used to calculate scores
#########################################################################################################

activation <- function(x){
  #tmp = NULL
  #eps = 0.0001
  #for(i in 1:length(x)){
  #  if(x[i]<0){
  #    tmp[i] = eps*x[i]
  #  } else{
  #    tmp[i]=x[i]
  #  }
  #}
  #return(tmp)
  value = tanh(x)
  return(value)
}

score <- function(x){
  #Uniform
  #rank = 1
  #Inverse
  rank = min(1/x,1000000)
  #exponential
  #rank = exp(-x)
  #rank = 1/cosh(x)
  return(rank)
}

mse <- function(x,y){
  temp = NULL
  for(i in 1:length(x)){
    temp[i] = (x[i]-y[i])^2
  }
  return(mean(temp))
}

estimated_curve <- function(x){
  return(rbinom(1,length(x),(cos(x)+1)/2))
}

probability <-function(x){
  if(x<0.3){
    return(0.05)
  }
  if(0.3<=x & x<0.6){
    return(0.25)
  }
  if(0.6<=x& x<0.8){
    return(0.75)
  }
  else{
    return(0.95)
  }
}

classes <- function(probs){
  return(rbinom(1,length(probs),probs))
}

nearestneighbours <- function(x,y,n){
  distances = abs(x-y)
  samples = NULL
  for (i in 1:n){
    samples[i] = which.min(distances)
    distances[samples[i]] = 10000000
  }
  return(samples)
}

main <- function(){
  #Architechture of neural network:
  number_of_inputs <<- 1
  number_of_hiddennodes1 <<- 100
  number_of_outputs <<- 1
  
  
  w1 <<- matrix(rnorm((number_of_inputs+1)*number_of_hiddennodes1,0,1), ncol = (number_of_inputs+1))
  w2 <<- matrix(rnorm((number_of_hiddennodes1+1)*number_of_outputs,0,1), ncol = (number_of_hiddennodes1+1))
  
  w1check <<- w1
  w2check <<- w2
  
  upperbound <<- 1
  lowerbound <<- 0
  nearestsamples <<- 40
  
  #Reseting the accuracy so that we can track it throughout the algorithm
  it <<- 1
  
  #Sample Size
  smp_size <<- 3000
  vld_size <<- 300
  
  #Size of sampling set:
  m <<- 150
  
  #Size of updating set:
  n <<- 150
  
  #Number of epochs:
  epochs <<- 50
  
  #Number of Trials:
  trials <<- 50
  
  xtrain <<- runif(smp_size,lowerbound,upperbound)
  ytrain <<- sapply(xtrain,probability)
  ytrain <<- sapply(ytrain,classes)
  
  valid_x <<- runif(vld_size,lowerbound,upperbound)
  valid_y <<- sapply(valid_x,probability)
  valid_y <<- sapply(valid_y,classes)
}

correct_log = matrix(NA, nrow = epochs, ncol=0)

for(trial in 1:trials){
  main()
  par(mfrow=c(1,2))
  correct = NULL
  
  for(epoch in 1:epochs){
    
    train_ind = sample(1:length(xtrain), size = smp_size/2, replace = FALSE)
    traindata = cbind(xtrain[train_ind],ytrain[train_ind])
    
    testdata = cbind(xtrain[-train_ind],ytrain[-train_ind])
    
    rows <- sample(nrow(testdata))
    
    testdata = testdata[rows,]
    
    for(k in 1:10){
      #These values move along the data sets so that we see new observations throughout the process
      a = 1 + m*(k-1)
      b = m*k
      c = 1 + n*(k-1)
      d = n*k
      data1 = traindata[a:b,]
      data2 = testdata[c:d,]
      
      #Tracking the observations and all their hidden layer values during the feed forward process.
      #This just saves us from having to calculate any inverses
      samplevalues = NULL
      
      for(i in 1:m){
        singlesample = NULL
        x1 = c(1,as.numeric(data1[i,1]))
        y1 = w1check %*% x1
        x2 = c(1,sapply(y1,activation))
        y2 = w2check %*% x2
        truey2 = as.numeric(data1[i,2])
        singlesample = c(x1,y1,y2,truey2)
        samplevalues = rbind(samplevalues,singlesample)
      }
      
      cat("sampled", "\n")
      
      for(i in 1:n){
        updatex1 = c(1,as.numeric(data2[i,1]))
        updatey2 = as.numeric(data2[i,2])
        nn = nearestneighbours(updatex1[2], samplevalues[,2],max(40-epoch,8))
        
        tmpvalues = samplevalues[nn,]
        
        errors = NULL
        for(y in 1:length(nn)){
          errors[y] = ((updatex1[2]-tmpvalues[y,2])^2+100*(updatey2-tmpvalues[y,(ncol(tmpvalues)-1)])^2)/2
        }
        tmpvalues = cbind(tmpvalues, errors)
        
        regularized = tmpvalues[,ncol(tmpvalues)]-min(tmpvalues[,ncol(tmpvalues)])
        probs = sapply(regularized,score)
        probs = probs/sum(probs)
        point = sample(1:length(nn),1,prob = probs)
        
        
        updatey1 = as.numeric(tmpvalues[point,(2+number_of_inputs):(number_of_inputs+number_of_hiddennodes1+1)]) 
        updatex2 = c(1,sapply(updatey1,activation))
        
        
        
        if(it==1){
          A1check = updatex1 %*% t(updatex1)
          A2check = updatex2 %*% t(updatex2)
          B1check = updatey1 %*% t(updatex1)
          B2check = updatey2 %*% t(updatex2)
          
        }
        
        if(it<16){
          gain1 = 1.5/max(eigen(A1check)$values)
          gain2 = 0.9/max(eigen(A2check)$values)
        }
        
        A1 = updatex1 %*% t(updatex1)
        A2 = updatex2 %*% t(updatex2)
        B1 = updatey1 %*% t(updatex1)
        B2 = updatey2 %*% t(updatex2)
        
        A1check = A1check +(A1 -A1check)/(it+1)
        B1check = B1check +(B1 -B1check)/(it+1)
        A2check = A2check +(A2 -A2check)/(it+1)
        B2check = B2check +(B2 -B2check)/(it+1)
        
        w1 = w1 + gain1*(B1check-w1%*%A1check)
        w2 = w2 + gain2*(B2check-w2%*%A2check)
        
        w1check = w1check + (w1-w1check)/(it+1)
        w2check = w2check + (w2-w2check)/(it+1)
        
        it=it+1
        
        if(i%%50==0){
          
          gain1 = 1.95/max(eigen(A1check)$values)
          gain2 = 1.95/max(eigen(A2check)$values)
        }
      }
      
    }
    
    outputs = NULL
    sum = 0
    for (s in 1:length(valid_x)){
      x1 = c(1,valid_x[s])
      X1tmp = w1check %*% x1
      x2 = c(1,sapply(X1tmp,activation))
      output = w2check %*% x2
      if(output >= 0.5){
        output=1
      }
      else{
        output=0
      }
      if(output==valid_y[s]){
        sum=sum+1
      }
      outputs=c(outputs,output)
    }
    correct = c(correct, sum/length(valid_x))
    plot(valid_x,valid_y,xlim = c(lowerbound,upperbound))
    points(valid_x,outputs, col="blue")
    #legend("topleft", legend = c("True Values", "NN Values"), col = c("black", "blue"), lty=1)
    plot(correct, type = "l", main = "Mean Squared Error for Dataset")
    
    
  }
  cat("Iteration: ", trial, " complete")
  correct_log = cbind(correct_log,correct)
}

round(rowMeans(correct_log),5)

probs = seq(-1,2,0.01)
TPR = NULL
FPR = NULL

for(v in 1:length(probs)){
  tp=0
  fp=0
  tn=0
  fn=0
  for(i in 1:vld_size){
    x1 = c(1,valid_x[i])
    X1tmp = w1check %*% x1
    x2 = c(1,sapply(X1tmp,activation))
    output = w2check %*% x2
    if(output >= probs[v]){
      output=1
    }
    else{
      output=0
    }
    if(output==1){
      if(valid_y[i]==1){
        tp=tp+1
      }
      else{
        fp=fp+1
      }
    }
    else{
      if(valid_y[i]==0){
        tn=tn+1
      }
      else{
        fn=fn+1
      }
    }
  }
  TPR = c(TPR,tp/(tp+fn))
  FPR = c(FPR,fp/(fp+tn))
}

TPR=c(TPR,0)
FPR=c(FPR,0)
par(mfrow=c(1,1))
plot(FPR,TPR,type="l",ylim=c(0,1),xlab="False Positive Rate", ylab = "True Positive Rate")

