% Define the PDE and boundary conditions
u = @(x, t) sin(pi*x).*exp(-t);
f = @(x, t) pi^2*sin(pi*x).*exp(-t);
bc = @(t) [0, 0];

% Define the domain and time range
x = linspace(0, 1, 100);
t = linspace(0, 1, 50);

% Create a meshgrid of (x, t) values
[X, T] = meshgrid(x, t);

% Flatten the (X, T) matrices into column vectors
xdata = X(:);
tdata = T(:);

% Generate the training data
u_train = u(xdata, tdata);
f_train = f(xdata, tdata);

% Define the neural network architecture
layers = [featureInputLayer(2)    fullyConnectedLayer(20)    reluLayer    fullyConnectedLayer(1)    regressionLayer];

% Define the training options
options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true);

% Train the neural network
net = trainNetwork([xdata, tdata], u_train, layers, options);

%%
u = @(x, t) cos(pi*x).*exp(-t);

% Evaluate the neural network on a test set
xtest = linspace(0, 1, 100);
ttest = linspace(0, 1, 100);
[Xtest, Ttest] = meshgrid(xtest, ttest);
utest = u(Xtest, Ttest);
ytest = predict(net, [Xtest(:), Ttest(:)]);
Ytest = reshape(ytest, size(Xtest));

% Plot the exact solution and neural network solution
figure;
subplot(1, 2, 1);
surf(Xtest, Ttest, utest);
xlabel('x');
ylabel('t');
zlabel('u(x,t)');
title('Exact solution');

subplot(1, 2, 2);
surf(Xtest, Ttest, Ytest);
xlabel('x');
ylabel('t');
zlabel('u(x,t)');
title('Neural network solution');


