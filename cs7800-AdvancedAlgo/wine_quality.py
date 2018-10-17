import cvxpy as cvx
import numpy as np
import csv
from cvxpy.atoms.affine.binary_operators import MulExpression


# Import the data from the .csv file
with open("wine_data.csv", newline='') as input_file: 
	# fieldnames = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
	reader = csv.DictReader(input_file, delimiter=';')
	data = []
	for row in reader: 
		data.append(row)

x = []
y = []

# Define the number of training samples
num_train_samples = 1500

# Format the training data into 1x1500 array y and 11x1500 matrix x 
for i in range(num_train_samples):

	train_row = list(data[i].values())

	y.append(float(train_row[len(train_row)-1]))

	train_row = np.array([train_row[0:len(train_row)-1]])
	train_row = np.transpose(train_row)

	x.append(train_row)

x = np.array(x).astype(float)
x = np.squeeze(x)
x = np.transpose(x)


# Create three optimization variables
a = cvx.Variable((1,11))
b = cvx.Variable()
z = cvx.Variable(num_train_samples)

# Create the LP constraints
constraints = []

for j in range(num_train_samples):
	constraints += [y[j]- MulExpression(a,x[:,j]) - b <= z[j], 
					y[j]- MulExpression(a,x[:,j]) - b >= -z[j]]


# Form the objective
obj = cvx.Minimize((1/num_train_samples)*sum(z))

# Form and solve the problem
prob = cvx.Problem(obj, constraints)
prob.solve()  

# Print the optimization results 
print("\nstatus:", prob.status)
print("optimal value", prob.value)
print("optimal var", a.value, b.value, "\n")

a_opt = a.value
b_opt = b.value

# Calculate the distance between the predicted score value (using optimization results) and the 
# actual score value for all of the training data
dist_train = np.absolute(y - np.matmul(a_opt,x) - b_opt)

# Calculate the average training error (both as a value and as a percent)
avg_training_err = (1/num_train_samples)*np.sum(dist_train)
avg_training_err_percent = (1/num_train_samples)*np.sum(np.divide(dist_train,y))*100

# Print the training error results
print("The average training error is %.8f" % avg_training_err)
print("The average training error percent is %.4f%%\n" % avg_training_err_percent)


x_test = []
y_test = []

# Format the test data into 1x99 array y and 11x99 matrix x 
for k in range(num_train_samples,len(data)):

	test_row = list(data[k].values())

	y_test.append(float(test_row[len(test_row)-1]))

	test_row = np.array([test_row[0:len(test_row)-1]])
	test_row = np.transpose(test_row)

	x_test.append(test_row)

x_test = np.array(x_test).astype(float)
x_test = np.squeeze(x_test)
x_test = np.transpose(x_test)

# Define the number of test samples
num_test_samples = len(y_test)

# Calculate the distance between the predicted score value (using optimization results) and the 
# actual score value for all of the test data
dist_test = np.absolute(y_test - np.matmul(a_opt,x_test) - b_opt)

# Calculate the average test error (both as a value and as a percent)
avg_test_err = (1/num_test_samples)*np.sum(dist_test)
avg_test_err_percent = (1/num_test_samples)*np.sum(np.divide(dist_test,y_test))*100

# Print the test error results
print("The average test error is %.8f" % avg_test_err)
print("The average test error percent is %.4f%%\n" % avg_test_err_percent)




