function [J grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),num_labels, (hidden_layer_size + 1));


Theta1_grad=zeros(size(Theta1));
Theta2_grad=zeros(size(Theta2));
m = size(X, 1);
ny=zeros(m,num_labels);
for i=1:m
    ny(i,y(i))=1;
end


a1=[ones(m,1) X];
z2=a1*(Theta1');
a2=[ones(m,1), sigmoid(z2) ];
z3=a2*(Theta2');
a3=sigmoid(z3);
J=(sum(sum(ny.*log(a3)+(1-ny).*log(1-a3))))/(-m)+(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

del3=a3-ny;
del2=(del3*Theta2).*(a2.*(1-a2));

Theta2_grad=(del3')*a2;
Theta2_grad=Theta2_grad./m;
Theta2_grad=Theta2_grad+ (lambda/m)*([zeros(size(Theta2,1),1),Theta2(:,2:end)]);
Theta1_grad=((del2(:,2:end))')*a1;
Theta1_grad=Theta1_grad./m;
Theta1_grad=Theta1_grad+ (lambda/m)*([zeros(size(Theta1,1),1),Theta1(:,2:end)]);



grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
