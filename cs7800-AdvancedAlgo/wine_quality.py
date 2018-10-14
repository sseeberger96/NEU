import cvxpy as cvx
import numpy as np
import csv
from cvxpy.atoms.affine.binary_operators import MulExpression



with open("wine_data.csv", newline='') as input_file: 
	# fieldnames = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
	reader = csv.DictReader(input_file, delimiter=';')
	data = []
	for row in reader: 
		data.append(row)

x = []
y = []

samples = 1500

for i in range(samples):

	k = list(data[i].values())

	y.append(float(k[len(k)-1]))

	k = np.array([k[0:len(k)-1]])
	k = np.transpose(k)
	# k = [float(i) for i in k]

	x.append(k)

# Create three optimization variables
a = cvx.Variable((1,11))
b = cvx.Variable()
z = cvx.Variable(samples)

# Create the constraints
constraints = []

for j in range(samples):
	constraints += [y[j]- MulExpression(a,x[j]) - b <= z[j], 
					y[j]- MulExpression(a,x[j]) - b >= -z[j]]


# Form the objective
obj = cvx.Minimize((1/samples)*sum(z))

# Form and solve problem.
prob = cvx.Problem(obj, constraints)
prob.solve()  

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", a.value, b.value)





