import numpy as np
import math
from numpy import linalg as LA

LAMBDA = 100
RUNS=15
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

def calculate_differential(matrix,betas):
	sum = 0
	for i in range(len(matrix)):
		p=np.dot(betas,matrix[i][:801])
		p=sigmoid(p)
		y=matrix[i][801]
		for j in range(len(matrix[i][:801])):
			x=matrix[i][j]
			sum += (y-p)*x
	return sum

def main():
	matrix=importdata(559,802,'/Users/suvir/Documents/Logistic Regression/train')
	#matrix=matrix[matrix[:,440].argsort()]
	#temp=np.ones(1,559)
	save_to_file("array.bin",matrix)
	betas=np.ones((1,801))
	betas[0][:400].fill(0.0125)
	betas[0][400:].fill(-0.0125)
	#Remember to randomize the dataset using random.randint(a,b)
	for runs in range(RUNS):
		print runs
		for i in range(len(matrix)):
			p=np.dot(betas,matrix[i][:801])
			p=sigmoid(p)
			y=matrix[i][801]
			for j in range(len(matrix[i][:801])):	
				x=matrix[i][j]
				betas[0][j] += LAMBDA*(y-p)*x
		#if runs%20 == 0:
	#		print runs
		print calculate_differential(matrix,betas)
	#print betas
	save_to_file("logistic_regression_beta_values_sorted.bin",betas)
	#calculate_label(matrix,betas)
#main()

matrix=importdata(239,802,'/Users/suvir/Documents/Logistic Regression/test')
for i in range(len(matrix)):
	matrix[i][:801]=np.divide(matrix[i][:801],LA.norm(matrix[i][:801]))
np.savetxt("test.txt",matrix)
#betas = load_from_file("experimental_betas.bin")
#print np.max(betas)
#print np.min(betas)
#print np.median(betas)

#matrix=importdata(559,802,'/Users/suvir/Documents/Logistic Regression/train')
#save_to_file("normalized_train.bin",matrix)
# count1=0
# count0=0
# matrix = load_from_file("train.bin")
# print np.max(matrix[0][:801])
# print np.min(matrix[0][:801])
# #matrix=importdata(239,802,'/Users/suvir/Documents/Logistic Regression/test')
# #save_to_file("test.bin",matrix)
# for j in range(len(matrix)):
# 	if matrix[j][801]==1:
# 		count1 += 1
# 	if matrix[j][801]==0:
# 		count0 += 1
# print count0,count1
# print float(count0)/len(matrix),float(count1)/len(matrix)
#matrix1=load_from_file("array.bin")
#Sanity Check
#print np.array_equal(matrix,matrix1)