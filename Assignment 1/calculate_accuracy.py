import numpy as np
import math
"""
Load array from file
"""
def load_from_file(filename):
	f = file(filename,"rb")
	matrix = np.load(f)
	return matrix

def calculate_accuracy(betas_filename,data_filename):
	#Test
	matrix=load_from_file(data_filename)
	#Train
	#matrix=load_from_file(data_filename)[:400]
	#Validation
	#matrix=load_from_file(data_filename)[401:]
	betas=load_from_file(betas_filename)
	count=0
	for i in range(len(matrix)):
		temp=np.dot(betas,matrix[i][:801])
<<<<<<< HEAD
		#temp = 1 / (1 + math.exp(-temp))
=======
		temp=1/(1+math.exp(-temp))
>>>>>>> 1348edeb118ed079ad0ec15cfd1700dbbb144ec5
		if temp >= 0.5 and matrix[i][801] == 1:
			count += 1
		if temp < 0.5 and matrix[i][801] == 0:
			count += 1
	print count
	print float(count)/float(len(matrix))

<<<<<<< HEAD
#Without regularization
#calculate_accuracy("logistic_regression_beta_values_sorted.bin","train.bin")
#calculate_accuracy("logistic_regression_beta_values_with_regularization_sortedOnY.bin","train.bin");
#Regularization
#calculate_accuracy("logistic_regression_beta_values_sorted.bin","train.bin")
#calculate_accuracy("logistic_regression_beta_values_with_regularization_sortedOnY.bin","train.bin");
#
#Test
#calculate_accuracy("logistic_regression_beta_values_sorted.bin","test.bin")
calculate_accuracy("logistic_regression_beta_values_with_regularization_sortedOnY.bin","test.bin");

=======
calculate_accuracy("logistic_regression_beta_values_with_regularization_sortedOnY.bin","normalized_test.bin")
>>>>>>> 1348edeb118ed079ad0ec15cfd1700dbbb144ec5
