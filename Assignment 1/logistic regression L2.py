import numpy as np
import math
from numpy import linalg as LA
<<<<<<< HEAD
LAMBDA = 0.02
LAMBDA0 = 1
C = 1.5
=======

LAMBDA = 0
LAMBDA0 = 0.02
C=1.5
E=1
>>>>>>> 1348edeb118ed079ad0ec15cfd1700dbbb144ec5
RUNS=100
MU = 1e-2
EPSILON = 0.001
TR = 1
DEL_BETA=0
LCL1=0
LCL2=0

"""
Import data from file to a numpy array
"""
def importdata(m,n,filename):
	matrix = np.zeros((m,n))
	f = open(filename)
	lines=f.readlines()
	f.close()
	for j in range(len(lines)):
		#if j%10==0:
		#	print j
		l=lines[j].strip().split(' ')
		for i in range(len(l)):
			matrix[j][0]=1
			if i == 0:
				if int(l[i]) == 1:
					matrix[j][801]=1
				else:
					matrix[j][801]=0
			else:
				col = int(l[i].split(':')[0])
				val = int(l[i].split(':')[1])
	 			matrix[j][col]=val

	#Normalize everything 
	for i in range(len(matrix)):
	 	matrix[i][:801]=np.divide(matrix[i][:801],LA.norm(matrix[i][:801]))
	return matrix

"""
Save the array to file
"""
def save_to_file(filename,matrix):
	f = file(filename,"wb")
	np.save(f,matrix)
	f.close()

"""
Load array from file
"""
def load_from_file(filename):
	f = file(filename,"rb")
	matrix = np.load(f)
	return matrix

def sigmoid(x):
	#print x
	return 1/(1+math.exp(-x))

def calculate_label(matrix,betas):
	count=0
	for i in range(len(matrix)):
		temp=np.dot(betas,matrix[i][:801])
		if temp==matrix[i][801]:
			count += 1
		#print temp
	#print count

def modify_lambda(n):
	#n is del RLCL
	global LAMBDA
	global LAMBDA0,C,E
	global TR
	global DEL_BETA
<<<<<<< HEAD

	LAMBDA = abs(DEL_BETA/n)


	# LAMBDA = abs(n)*0.0001
	print LAMBDA
	


=======
	#LAMBDA = DEL_BETA/DEL_LCL
	# LAMBDA = abs(n)*0.0001
	# print LAMBDA
>>>>>>> 1348edeb118ed079ad0ec15cfd1700dbbb144ec5
	# if abs(n)<TR:
	# 	LAMBDA = LAMBDA*0.2
	# 	TR = float(TR)/5
	# 	print "LAMBDAAAAAAAA"
	# 	print LAMBDA
	# 	print TR
	LAMBDA = LAMBDA0/(1+100*LAMBDA0*C*E)

def calculate_differential(matrix,betas):
	global LCL1
	global LCL2
	sum1 = 0
	sum2 = 0
	count = 0
	for i in range(len(matrix)):
		p=np.dot(betas,matrix[i][:801])
		p=sigmoid(p)
		y=matrix[i][801]
		for j in range(len(matrix[i][:801])):
			x=matrix[i][j]
			sum1 += (y-p)*x
			#sum1 is del(RLCL)
			#sum2 is LCL
		sum1 = sum1 - 2*MU*betas[0][j]		
		sum2 += y*math.log(p+0.000001)+(1-y)*math.log(1-p+0.000001)
		if p < 0.5 and y == 0 :
			count += 1
		if p >= 0.5 and y == 1 :
			count += 1
		
	return sum1, sum2, float(count) / len(matrix)
def calculateValidationAccuracy(matrix, betas):
	count = 0
	for i in range(len(matrix)):
		p=np.dot(betas,matrix[i][:801])
		p=sigmoid(p)
		y=matrix[i][801]
		if p < 0.5 and y == 0 :
			count += 1
		if p >= 0.5 and y == 1 :
			count += 1
	return float(count) / len(matrix)

def main():
<<<<<<< HEAD
	matrix=importdata(559,802,'train')
=======
	global E
	matrix=importdata(559,802,'/Users/suvir/Documents/Logistic Regression/train')
>>>>>>> 1348edeb118ed079ad0ec15cfd1700dbbb144ec5
	print "Started sorting"
	matrix_full = matrix[matrix[:,600].argsort()]
	matrix = matrix_full[:400]
	print "Finished sorting"
	#temp=np.ones(1,559)
	save_to_file("array.bin",matrix)
	betas=np.ones((1,801))
<<<<<<< HEAD
	betas[0][:400].fill(0)
	betas[0][400:].fill(0)
	lcl = 10
	lastLcl = 0
	DEL_RLCL = 10
	lastValAccuracy = 0
	#Remember to randomize the dataset using random.randint(a,b)
	global LAMBDA
	runs = 0
	#for runs in range(RUNS):
	while abs(lcl - lastLcl) > EPSILON:
		DEL_BETA = 0
=======
	betas[0][:400].fill(0.002)
	betas[0][400:].fill(0.002)
	#Remember to randomize the dataset using random.randint(a,b)
	for runs in range(RUNS):
		#print runs
		E = runs
>>>>>>> 1348edeb118ed079ad0ec15cfd1700dbbb144ec5
		for i in range(len(matrix)):
			p=np.dot(betas,matrix[i][:801])
			p=sigmoid(p)
			y=matrix[i][801]
			for j in range(len(matrix[i][:801])):
				x=matrix[i][j]
				betas[0][j] += LAMBDA*((y-p)*x-2*MU*betas[0][j])
		lastLcl = lcl
		DEL_RLCL, lcl, trainAccuracy = calculate_differential(matrix,betas)
		valAccuracy = calculateValidationAccuracy(matrix_full[401:], betas)
		if lastValAccuracy < valAccuracy:
			lastValAccuracy = valAccuracy
			bestBetas = betas
		print runs, DEL_RLCL, lcl
		runs += 1
		#LAMBDA = LAMBDA0 / (1 + runs * C)
	#print betas
	save_to_file("logistic_regression_beta_values_with_regularization_sortedOnY.bin",bestBetas)
	#calculate_label(matrix,betas)i
main()
