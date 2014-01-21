import numpy as np
"""
Load array from file
"""
def load_from_file(filename):
	f = file(filename,"rb")
	matrix = np.load(f)
	return matrix

def calculate_accuracy(betas_filename,data_filename):
	matrix=load_from_file(data_filename)
	betas=load_from_file(betas_filename)
	count=0
	for i in range(len(matrix)):
		temp=np.dot(betas,matrix[i][:801])
		if temp >= 0.5 and matrix[i][801] == 1:
			count += 1
		if temp < 0.5 and matrix[i][801] == 0:
			count += 1
	print float(count)/float(len(matrix))

calculate_accuracy("logistic_regression_beta_values_sorted.bin","test.bin")
