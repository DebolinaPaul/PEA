library(matlib)
pea_projection=function(Y,y_new){
  n=dim(Y)[1]
  p=dim(Y)[2]
  Z=matrix(0,n,p)
  Y_bar=colMeans(Y)
  for(i in 1:n){
    Z[i,]=Y[i,]-Y_bar
  }
  U=numeric(n)
  for(i in 1:n){
    U[i]=sum(Y[i,]^2)
  }
  U_centre=U-mean(U)
  nS=(n-1)*var(Z)
  temp=numeric(p)
  for(i in 1:n){
    temp=temp+U_centre[i]*Z[i,]
  }
  InS=inv(nS)
  b=numeric(p)
  b=InS%*%(temp)
  c=-(sum(b*Y_bar)/n)-mean(U)
  mu=-b/2
  r=sqrt(sum(mu^2)-c)
  Y_proj=matrix(0,n,p)
  #  for(i in 1:n){
  #    Y_proj[i,]=mu+(Y[i,]-mu)*r/sqrt(sum((Y[i,]-mu)^2))
  #  }
  y_proj=mu+(y_new-mu)*r/sqrt(sum((y_new-mu)^2))
  return(list(mu,r,y_proj))
}
fff=function(x){
  p=length(x)
  a=sqrt(sum(x^2))
  if(a<10^(-10)){
    return(numeric(p))
  }else{
    return(x/a)
  }
}
proj_new=function(x_new,w,mu,U){
  p=dim(U)[2]
  ss=numeric(p)
  y=numeric(p)
  for(l in 1:p){
    if(sum(U[,l]^2)>10^(-4)){
      ss[l]=1
    }
  }
  if(sum(ss)>0){
    I=which(ss==1)
    a=(x_new[I]-mu[I])
    b=sqrt(sum(a^2))
    a=a/b*w+mu[I]
    y[I]=a
  }
  return(y)
}

proj_new_new=function(x_new,U,X){
  p=dim(U)[2]
  ss=numeric(p)
  y=numeric(p)
  for(l in 1:p){
    if(sum(U[,l]^2)>10^(-4)){
      ss[l]=1
    }
  }
  if(sum(ss)>0){
    I=which(ss==1)
    alu=pea_projection(X,x_new)
    y[I]=alu[[3]]
  }
  return(y)
}
f=function(x){
  return((x/sqrt(sum(x^2))))
}
S=function(x,y){
  a=abs(x)
  if(x<y){
    return(0)
  }else{
    return(sign(x)*(a-y))
  }
}
sparse_spherical_pca=function(X,lambda,tmax=100){
  # Code with penalizer
  n=dim(X)[1]
  p=dim(X)[2]
  U=X
  #  for(i in 1:p){
  #    X[,i]=(X[,i]-mean(X[,i]))/sd(X[,i])
  #  }
  U=X
  mu=colMeans(X)
  #mu=rep(-30,4)
  U=apply(X-mu, 1, fff)
  U=t(U)
  for(t in 1:tmax){
    # Update w
    w=sum(X*U)
    for(l in 1:p){
      w=w-sum(U[,l])*mu[l]
    }
    w=w/n
    # Update mu
    for(l in 1:p){
      mu[l]=mean(X[,l]-w*U[,l])
    }
    #update U
    for(l in 1:p){
      U[,l]=w*(X[,l]-mu[l])-lambda*fff(U[,l])
    }
    for(i in 1:n){
      U[i,]=fff(U[i,])
    }
    #    if(t%%1==0){
    #      cat(t)
    #      cat('\t')
    #      cat(w)
    #      cat('\n')
    #    }
  }
  Y=X
  for(i in 1:n){
    Y[i,]=U[i,]*w+mu
  }
  return(list(Y,w,mu,U))
}
euc.norm=function(x){
  return(sqrt(sum(x^2)))
}
feature.extraction=function(X,epsilon){
  n=dim(X)[1]
  p=dim(X)[2]
  Index=numeric(p)
  for(j in 1:p){
    a=euc.norm(X[,j])
    if(a>epsilon)
      Index[j]=1
    else
      Index[j]=0
  }
  return(Index)
}
d_sp=function(x,mu){
  return(acos(sum(x*mu)))
}
f=function(X,mu){
  # X is an n*p matrix
  # mu is an P*1 vector
  n=dim(X)[1]
  s=0
  for(i in 1:n){
    s=s+d_sp(X[i,],mu)
  }
  return(s)
}
grad_f=function(X,mu){
  n=dim(X)[1]
  p=dim(X)[2]
  s=numeric(p)
  for(i in 1:n){
    s=s-1/sqrt(1-(sum(X[i,]*mu))^2)*X[i,]
  }
  return(s)
}
my_frechet=function(X,eta,tmax,epsa){
  mu=colMeans(X)
  mu=fff(mu)
  a=f(X,mu)
  b=0
  for(t in 1:tmax){
    mu=mu-eta*grad_f(X,mu)
    mu=fff(mu)
    b=f(X,mu)
    if((a-b)<epsa){
      break
    }else{
      a=b
    }
    cat(b)
    cat('\n')
  }
  return(mu)
}

#Data Generator
Y=matrix(0,300,1000)
Y[1:100,1]=rnorm(100,0,0.5)
Y[101:200,1]=rnorm(100,5,0.5)
Y[1:100,2]=rnorm(100,0,0.5)
Y[101:200,2]=rnorm(100,5,0.5)
Y[201:300,1]=rnorm(100,10,0.5)
Y[201:300,2]=rnorm(100,0,0.5)
for(j in 3:1000){
  Y[,j]=rnorm(300,0,4)
}
toss=c(rep(1,100),rep(2,100),rep(3,100))
plot(Y,col=toss)
#Normalization
Y=X
p=dim(Y)[2]
for(j in 1:p){
  Y[,j]=(Y[,j]-mean(Y[,j]))/sd(Y[,j])
}

kmin=kmeans(Y,2)
1-compare(toss,kmin$cluster,method="rand")

#Running The Code
pea_clust=function(Y,k,lambda,epsa=3,tmax=300){
  l=sparse_spherical_pca(Y,lambda,tmax=300)
  X=l[[4]]
  Index=feature.extraction(X,epsa)
  I=which(Index==1)
  U=X[,I]
  d=dim(U)[2]
  for(i in 1:d){
    U[,i]=U[,i]/euc.norm(U[,i])
  }
  U_0=my_frechet(U,0.01,1000,10^(-8))
  Z=U
  n=dim(U)[1]
  for(i in 1:n){
    a=(U[i,]-sum(U[i,]*U_0)*U_0)
    Z[i,]=a*acos(sum(U[i,]*U_0))/sqrt(sum(a^2))+U_0
  }
  kmin=kmeans(Z,k,iter.max = 200)
  return(kmin)
}
l=sparse_spherical_pca(X,0.01,tmax=2000)
X1=l[[4]]
sqrt(colSums(X1^2))

plot(X1,col=label)
Index=feature.extraction(X1,6.22)
I=which(Index==1)
I
U=X1[,I]
d=dim(U)[2]
for(i in 1:d){
  U[,i]=U[,i]/euc.norm(U[,i])
}
plot(U)
U_0=my_frechet(U,0.01,1000,10^(-8))
Z=U
n=dim(U)[1]
for(i in 1:n){
  a=(U[i,]-sum(U[i,]*U_0)*U_0)
  Z[i,]=a*acos(sum(U[i,]*U_0))/sqrt(sum(a^2))+U_0
}
plot(Z,col=label)
kmin=kmeans(Z,2,iter.max = 200)
1-compare(toss,kmin$cluster,method = "rand")
compare(toss,kmin$cluster,'nmi')
CER=function(l1,l2){
  n=length(l1)
  s=0
  for(i in 1:(n-1)){
    for(j in (i+1):n){
      s=s+abs((l1[i]==l1[j])-(l2[i]==l2[j]))
    }
    return(s/(n*(n-1))*2)
  }
}
CER(toss,kmin$cluster)
CER <- function(clusters, classes) {
  sum(apply(table(classes, clusters), 2, max)) / length(clusters)
}
CER(toss,kmin$cluster)
####### Real testing
###_______________WDBC__________############
X=read.csv('C:\\Users\\User-PC\\Desktop\\Datasets\\classification keel\\wdbc.csv',head=FALSE)
dim(X)
X=data.matrix(X)
toss=X[,31]
X=X[,-31]
l=pea_clust(X,lambda=200,epsa=3)
compare(l$cluster,toss,'adjusted.rand')
compare(l$cluster,toss,'nmi')

###_______________Mammographic__________############
X=read.csv('C:\\Users\\User-PC\\Desktop\\Datasets\\classification keel\\mammographic.csv',head=FALSE)
dim(X)
X=data.matrix(X)
toss=X[,6]
X=X[,-6]
l=pea_clust(X,k=2,lambda=100,epsa=2)
compare(l$cluster,toss,'adjusted.rand')
compare(l$cluster,toss,'nmi')

###_______________Brain__________############
X=read.table('C:\\Users\\User-PC\\Desktop\\Datasets\\microarray\\brain_x.txt',head=FALSE)
toss=read.table('C:\\Users\\User-PC\\Desktop\\Datasets\\microarray\\brain_y.txt',head=FALSE)
X=data.matrix(X)
X=t(X)
toss=c(data.matrix(toss))
l=pea_clust(X,k=5,lambda=10,epsa=0.1)
compare(l$cluster,toss,'adjusted.rand')
compare(l$cluster,toss,'nmi')

D=matrix(0,n,n)
for(i in 1:n){
  for(j in 1:n){
    D[i,j]=sqrt(sum((X[i,]-X[j,])^2))
  }
}

h=hclust(D,'single')
h=hclust(dist(X))
?dist
dist(X[1:3,])

arc_cos
as.dist(D[1:3,1:3])
pea_clust_hc=function(Y,lambda,epsa=3,tmax=300){
  n=dim(Y)[1]
  l=sparse_spherical_pca(Y,lambda,tmax=300)
  X=l[[4]]
  Index=feature.extraction(X,epsa)
  I=which(Index==1)
  U=X[,I]
  d=dim(U)[2]
  for(i in 1:d){
    U[,i]=U[,i]/euc.norm(U[,i])
  }
  D=matrix(0,n,n)
  for(i in 1:n){
    for(j in 1:n){
      D[i,j]=acos(sum(U[i,]*U[j,]))
    }
  }
  D=as.dist(D)
  h=hclust(D,'complete')
  return(h)
}
l=pea_clust_hc(X,lambda=5,epsa=6.2)
plot(l)
library(sparcl)
ColorDendrogram(l,label)
km=kmeans(X[,1:3],11)
library(igraph)
compare(label,km$cluster,'adjusted.rand')
