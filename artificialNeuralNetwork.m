function artificialNeuralNetwork



    clc; clear; close all;
    rng(42);              
    
    n_samples = 100; 

    mean0 = [-1, -1];
    mean1 = [ 1,  1];
    cov_mat = [0.5 0; 0 0.5];

    X0 = mvnrnd(mean0, cov_mat, n_samples);
    X1 = mvnrnd(mean1, cov_mat, n_samples);

    X = [X0; X1];               
    y_class = [zeros(n_samples,1); ones(n_samples,1)];  

    
    
   
    Y = zeros(2, length(y_class));
    Y(1, y_class==0) = 1;
    Y(2, y_class==1) = 1;

    
    X = X';       

   
    layer_sizes = [2, 3, 2];
    nn = init_network(layer_sizes);

    
    fprintf('--- Kısa forward/backward testleri (mock) ---\n');
    mock_X = [-0.4838731  0.08083195; ...
               0.93456167 -0.50316134];      
    mock_X = mock_X';
    mock_X = mock_X';                        

    mock_Y = [1 0; 0 1];                     
    [cache_mock, mock_hat] = forward_prop(nn, mock_X);
    loss_mock = compute_loss(mock_Y, mock_hat);
    fprintf('Mock forward loss (sadece kontrol amaçlı): %.4f\n', loss_mock);

    grads_mock = back_prop(nn, cache_mock, mock_Y);
    fprintf('Mock backward gradyanları hesaplandı.\n');

    
    fprintf('\n--- Gradient check ---\n');
    gradient_check(nn, X(:,1:5), Y(:,1:5));  

  
    fprintf('\n--- Eğitime başlıyoruz ---\n');
    num_epochs    = 2000;
    learning_rate = 0.1;

    [nn, loss_hist] = train_network(nn, X, Y, num_epochs, learning_rate);

   
    fprintf('Son loss: %.4f\n', loss_hist(end));

    
    [~, Y_hat] = forward_prop(nn, X);
    [~, pred_class] = max(Y_hat, [], 1);
    pred_class = pred_class - 1;      % 1->0, 2->1
    acc = mean(double(pred_class' == y_class));
    fprintf('Eğitim doğruluğu: %.2f %%\n', acc*100);

    
    figure; hold on;
    scatter(X0(:,1), X0(:,2), 20, 'b', 'filled');
    scatter(X1(:,1), X1(:,2), 20, 'r', 'filled');

    
    x1_min = min(X(1,:)) - 1; x1_max = max(X(1,:)) + 1;
    x2_min = min(X(2,:)) - 1; x2_max = max(X(2,:)) + 1;
    [xx1, xx2] = meshgrid(linspace(x1_min,x1_max,200), ...
                          linspace(x2_min,x2_max,200));
    grid_X = [xx1(:)'; xx2(:)'];

    [~, grid_hat] = forward_prop(nn, grid_X);
    [~, grid_pred] = max(grid_hat, [], 1);
    grid_pred = reshape(grid_pred, size(xx1));

    contourf(xx1, xx2, grid_pred, [0.5 1.5 2.5], 'LineColor','none','FaceAlpha',0.3);
    colormap([0.6 0.6 1; 1 0.6 0.6]);

    title('2-3-2 Yapay Sinir Ağı - Karar Sınırı');
    xlabel('x_1'); ylabel('x_2');
    legend('Sınıf 0','Sınıf 1');
    hold off;
end


function nn = init_network(layer_sizes)


    L = numel(layer_sizes)-1;   

    nn = struct;
    nn.L = L;
    nn.sizes = layer_sizes;
    nn.W = cell(L,1);
    nn.b = cell(L,1);

    for l = 1:L
        in_size  = layer_sizes(l);
        out_size = layer_sizes(l+1);

        
        nn.W{l} = randn(out_size, in_size) * sqrt(2/(in_size + out_size));
        nn.b{l} = zeros(out_size, 1);
    end
end

function [cache, A_L] = forward_prop(nn, X)


    L = nn.L;
    m = size(X,2);

    A = cell(L+1,1);
    Z = cell(L,1);

    A{1} = X;   

    for l = 1:L
        W = nn.W{l};
        b = nn.b{l};

        Z{l} = W * A{l} + repmat(b,1,m);

        if l < L
           
            A{l+1} = sigmoid(Z{l});
        else
           
            A{l+1} = softmax(Z{l});
        end
    end

    cache = struct;
    cache.A = A;
    cache.Z = Z;
    A_L = A{end};
end

function loss = compute_loss(Y, Y_hat)

    m = size(Y,2);
    eps_val = 1e-8;
    loss = - (1/m) * sum(sum(Y .* log(Y_hat + eps_val)));
end

function grads = back_prop(nn, cache, Y)


    L = nn.L;
    A = cache.A;
    Z = cache.Z;
    m = size(Y,2);

    grads.dW = cell(L,1);
    grads.db = cell(L,1);

    dZ = cell(L,1);
    dZ{L} = A{L+1} - Y;    

    
    grads.dW{L} = (1/m) * dZ{L} * A{L}';
    grads.db{L} = (1/m) * sum(dZ{L}, 2);

    
    for l = L-1:-1:1
        W_next = nn.W{l+1};
        dA = W_next' * dZ{l+1};          

        
        s = sigmoid(Z{l});
        dZ{l} = dA .* s .* (1-s);

        grads.dW{l} = (1/m) * dZ{l} * A{l}';
        grads.db{l} = (1/m) * sum(dZ{l}, 2);
    end
end

function [nn, loss_hist] = train_network(nn, X, Y, num_epochs, lr)


    loss_hist = zeros(num_epochs,1);

    for epoch = 1:num_epochs
        
        [cache, Y_hat] = forward_prop(nn, X);

        
        loss = compute_loss(Y, Y_hat);
        loss_hist(epoch) = loss;

        
        grads = back_prop(nn, cache, Y);

        
        for l = 1:nn.L
            nn.W{l} = nn.W{l} - lr * grads.dW{l};
            nn.b{l} = nn.b{l} - lr * grads.db{l};
        end

        if mod(epoch, 200) == 0
            fprintf('Epoch %4d / %4d, loss = %.4f\n', epoch, num_epochs, loss);
        end
    end
end

function gradient_check(nn, X, Y)



    epsilon = 1e-5;

    
    [cache, Y_hat] = forward_prop(nn, X);
    grads = back_prop(nn, cache, Y);
    dW1_analytic = grads.dW{1};

    
    dW1_numeric = zeros(size(dW1_analytic));

    W1 = nn.W{1};

    for i = 1:size(W1,1)
        for j = 1:size(W1,2)
            W_pos = W1;  W_neg = W1;
            W_pos(i,j) = W_pos(i,j) + epsilon;
            W_neg(i,j) = W_neg(i,j) - epsilon;

            nn_pos = nn; nn_neg = nn;
            nn_pos.W{1} = W_pos;
            nn_neg.W{1} = W_neg;

           
            [~, Y_hat_pos] = forward_prop(nn_pos, X);
            [~, Y_hat_neg] = forward_prop(nn_neg, X);

            L_pos = compute_loss(Y, Y_hat_pos);
            L_neg = compute_loss(Y, Y_hat_neg);

            dW1_numeric(i,j) = (L_pos - L_neg) / (2*epsilon);
        end
    end

    
    diff = norm(dW1_numeric - dW1_analytic) / ...
          (norm(dW1_numeric) + norm(dW1_analytic) + 1e-8);

    fprintf('Gradient check (katman 1, relative diff): %.2e\n', diff);
end



function s = sigmoid(z)
    s = 1 ./ (1 + exp(-z));
end

function A = softmax(Z)

    Z_shift = Z - max(Z,[],1);
    expZ = exp(Z_shift);
    A = expZ ./ sum(expZ, 1);
end
