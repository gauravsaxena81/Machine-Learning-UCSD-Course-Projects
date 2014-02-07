#from __future__ import print_function
import numpy as np
from numpy import linalg as LA
from scipy import linalg, optimize
from sklearn import cross_validation, metrics, linear_model
from datetime import datetime
import math
import sys

def save_to_file(filename,matrix):
	f = file(filename,"wb")
	np.save(f,matrix)
	f.close()

def sigmoid(x):
	#print x
	return 1/(1+math.exp(-x))

def gradient(w, X, y, alpha):
	gradient_vector=np.ones((1,801))
	#print w.shape
	for j in range(w.shape[0]):
		sum=0
		for i in range(X.shape[0]):
			Yi = y[0][i]
			Pi=np.dot(w,X[i])
			Pi=sigmoid(Pi)
			Xij=X[i][j]
			sum += (Yi - Pi)*Xij - 2*alpha*w[j] 
		gradient_vector[0][j]=sum
	#print gradient_vector.shape
	return gradient_vector.T

def RLCL(w,X,y,alpha):
	#This is the function to be minimized
	total_rlcl = 0
	for i in range(X.shape[0]):
		Yi = y[0][i]
		tPi=np.dot(w,X[i])
		Pi=sigmoid(tPi)
		try:
			total_rlcl += Yi*math.log(Pi) + (1-Yi)*math.log(1-Pi) - alpha*np.dot(w,w.T)
		except ValueError as e:
			save_to_file("lbfgs_betas.bin",w)
			sys.exit("Bye Bye World")
	print -total_rlcl
	return -total_rlcl

"""
Import data from file to a numpy array
"""
def importdata(m,n,filename):
	matrix = np.zeros((m,n))
	Y=np.zeros((1,559))
	f = open(filename)
	lines=f.readlines()
	f.close()
	for j in range(len(lines)):
		l=lines[j].strip().split(' ')
		for i in range(len(l)):
			#Intercept
			matrix[j][0]=1
			#Y matrix has the labels
			if i == 0:
				if int(l[i])==1:
					Y[0][j]=1
				else:
					Y[0][j]=0
			else:
				col = int(l[i].split(':')[0])
				val = int(l[i].split(':')[1])
	 			matrix[j][col]=val

	#Normalize everything 
	for i in range(len(matrix)):
	 	matrix[i]=np.divide(matrix[i],LA.norm(matrix[i]))
	print "Started sorting"
	matrix=matrix[matrix[:,600].argsort()]
	print "Finished sorting"
	return matrix,Y

def main():
	X,y=importdata(559,801,'/Users/suvir/Documents/Logistic Regression/train')
	w=np.ones((1,801))
	w[0].fill(5)
	#print w.shape
	alpha=0
	x,f,d = optimize.fmin_l_bfgs_b(RLCL, w, fprime=gradient, args=(X, y, alpha),approx_grad=0, 
		bounds=None, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-05, iprint=0, maxfun=15000, 
		maxiter=2000, disp=None, callback=None)
	save_to_file("lbfgs_betas.bin",x)
	#print f
	#print d['warnflag']
	#print d['grad']
main()

#X,y=importdata(559,801,'/Users/suvir/Documents/Logistic Regression/train')