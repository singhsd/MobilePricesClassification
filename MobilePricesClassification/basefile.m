
clear ; close all; clc

ip  = 20;  
hide = 18;   
labels = 4;      
                          
load('data.txt');
x=data(:,1:20); % see this
x=featureNormalize(x);
y=data(:,21);
m=size(x,1);

load('testme.txt');
testx=testme(:,1:20);
testx=featureNormalize(testx);

theta1=randInit(ip,hide);
theta2=randInit(hide,labels);
nn_params=[theta1(:) ; theta2(:) ];
for i=1:m
  if y(i)==0
    y(i)=4;
   end;
end;

lambda=1.5;
%while true,

options = optimset('MaxIter', 200);
costFunction = @(p) nnCostFunction(p,ip,hide,labels,x,y,lambda);

[new_params,cost]=fmincg(costFunction,nn_params,options);
theta1=reshape(new_params(1:hide*(ip+1)),hide,ip+1);
theta2=reshape(new_params((1+hide*(ip+1)):end),labels,hide+1);

guess=predict(theta1,theta2,testx);  % see this too
save submit.csv guess
