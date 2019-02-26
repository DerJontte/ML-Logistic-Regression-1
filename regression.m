function regression(training_size = 50)
  # Load relevant case data from file
  [reason, gender, age, mobility, distance, part] = textread("data.csv", "%s %s %s %s %s %s", "delimiter", ";");
  age = normalize(str2double(strrep(age, ",", ".")))(2:end);
  distance = normalize(str2double(strrep(distance, ",", ".")))(2:end);
  mobility = strcmp(mobility, "Car")(2:end);
  gender = strcmp(gender, "M")(2:end);
  part = str2double(part)(2:end);
  
  # Create appropriate vectors and matrices:
  #  - X holds the independent variables casewise
  #  - y holds the dependent variables casewise
  #  - theta holds the regression coefficients

  #  X = [ones(length(age), 1) [age distance mobility gender]]; -> 73,33%
  #  X = [ones(length(age), 1) [distance mobility gender]]; -> 72,67%
  #  X = [ones(length(age), 1) [age mobility gender]]; -> 69,33%
  #  X = [ones(length(age), 1) [age distance gender]]; -> 68,44%
  X = [ones(length(age), 1) [age distance mobility]]; # -> 77,56%
  # X = [ones(length(age), 1) [age distance]]; -> 71,11%
  # X = [ones(length(age), 1) [age mobility ]]; -> 69,56%
  # X = [ones(length(age), 1) [distance mobility ]]; -> 75,33%
  Y = part;
  theta = zeros(1, size(X, 2));

  # Create a training set. The default size of 50 samples can be overridden
  # by passing an argument to the program.
  training_x = X(1:training_size, :);
  training_y = Y(1:training_size, :);

  # Create a test set that is disjoint to the training set
  test_x = X(training_size + 1:end, :);
  test_y = Y(training_size + 1:end, :);
  test_size = length(test_y);
  
  disp(sprintf("Training set size: %d", training_size))
  disp(sprintf("Test set size: %d", test_size))
  
  # Run the gradient descent algorithm on the training samples
  theta = run_gradient(training_x, training_y, theta);
  disp("Calculated regression coefficients:")
  disp(theta)

  # Test the model against the entire dataset and print out how well it did
  predicted = predict(theta, test_x);
  correct = sum(predicted == test_y');
  
  disp(sprintf("Number of correct predictions: %d/%d (%.02f%%)", correct, test_size, (correct/test_size)*100))
endfunction

function retval = rss()
  # Function to calculate the residual sum of squares
    
endfunction
  
function retval = normalize(x)
  # Function to normalize the values in a vector
  min = min(x);
  max = max(x);
  range = max - min;
  retval = 1 - ((max .- x) ./ range);
endfunction

function retval = sigmoid(theta, Xi)
  # The sigmoid function
  retval = 1.0 / (1 + exp(-dot(theta', Xi)));
endfunction

function retval = gradient(X, y, theta)
  # Function to calculate the gradient
  for j = 1:length(theta)
    retval(j) = 0;
    for i = 1:length(y)
      h = sigmoid(theta, X(i,:));
      retval(j) += (h - y(i)) * X(i, j);
    endfor
  endfor
endfunction

function retval = cost_function(X, y, theta)
  # Function that calculates the cost function for a given set of regression coefficients
  retval = 0;
  for i = 1:length(y)
    h = sigmoid(theta, X(i,:));
    retval += -(y(i) * log(h)) - ((1.0 - y(i)) * log(1.0 - h));
  endfor
endfunction

function retval = run_gradient(X, y, theta, learn = 0.01, converge = 0.001)
  # The main function of the algorithm. It updates the regression coefficients until
  # the difference in cost between two updates are below the convergence value.
  # It then returns the regression coefficients that are to be used in the regression model.
  cost = cost_function(X, y, theta);
  cost_delta = 1;
  iterations = 0;

  while (cost_delta > converge)
    old_cost = cost;
    theta = theta - (learn * gradient(X, y, theta));
    cost = cost_function(X, y, theta);
    cost_delta = old_cost - cost;
    iterations += 1;
  endwhile
  
  disp(sprintf("Number of iterations: %d", iterations))
  retval = theta;
endfunction

function retval = predict(theta, X)
  # This function applies the regression to a matrix
  for i = 1:length(X)
    pred_prob(i) = sigmoid(theta, X(i,:));
  endfor
  retval = pred_prob >= 0.5;
endfunction