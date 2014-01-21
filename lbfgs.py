from __future__ import print_function
import numpy as np
from numpy import linalg as LA
from scipy import linalg, optimize
from sklearn import cross_validation, metrics, linear_model
from datetime import datetime
import math

LAMBDA = 100
RUNS=15

def phi(t):
    # logistic function
    idx = t > 0
    out = np.empty(t.size, dtype=np.float)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    #try:
    print(exp_t)
    out[~idx] = exp_t / (1. + exp_t)
    # #except:
    
    # print out.shape
    # print e
    return out

def loss(w, X, y, alpha):
    # loss function to be optimized, it's the logistic loss
    z = X.dot(w)
    yz = y * z
    idx = yz > 0
    out = np.zeros_like(yz)
    out[idx] = np.log(1 + np.exp(-yz[idx]))
    out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
    out = out.sum() + .5 * alpha * w.dot(w)
    return out

def gradient(w, X, y, alpha):
    # gradient of the logistic loss
    z = X.dot(w)
    
    z = phi(y * z)
    
    z0 = (z - 1) * y
    grad = X.T.dot(z0) + alpha * w
    return grad

def hessian(s, w, X, y, alpha):
    z = X.dot(w)
    z = phi(y * z)
    d = z * (1 - z)
    wa = d * X.dot(s)
    Hs = X.T.dot(wa)
    out = Hs + alpha * s
    return  out

def grad_hess(w, X, y, alpha):
    # gradient AND hessian of the logistic
    z = X.dot(w)
    z = phi(y * z)
    z0 = (z - 1) * y
    grad = X.T.dot(z0) + alpha * w
    def Hs(s):
        d = z * (1 - z)
        wa = d * X.dot(s)
        return X.T.dot(wa) + alpha * s
    return grad, Hs

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
				Y[0][j]=int(l[i])
			else:
				col = int(l[i].split(':')[0])
				val = int(l[i].split(':')[1])
	 			matrix[j][col]=val

	#Normalize everything 
	for i in range(len(matrix)):
	 	matrix[i]=np.divide(matrix[i],LA.norm(matrix[i]))

	return matrix,Y

def main():
	X,y=importdata(559,801,'/Users/suvir/Documents/Logistic Regression/train')
	w=np.ones((1,800))
	w[0][:400].fill(0.0125)
	w[0][400:].fill(-0.0125)
	alpha=1
	#print Y[0]

	# conjugate gradient
	print('CG')
	timings_cg = []
	precision_cg = []
	w0 = np.zeros(X.shape[1])
	start = datetime.now()
	def callback(x0):
	    prec = linalg.norm(gradient(x0, X, y, alpha), np.inf)
	    precision_cg.append(prec)
	    timings_cg.append((datetime.now() - start).total_seconds())
	callback(w0)
	w = optimize.fmin_cg(loss, w0, fprime=gradient, 
	        args=(X, y, alpha), gtol=1e-6, callback=callback, maxiter=200)

	## truncated newton
	print('TNC')
	timings_tnc = []
	precision_tnc = []
	w0 = np.zeros(X.shape[1])
	start = datetime.now()
	def callback(x0):
	    prec = linalg.norm(gradient(x0, X, y, alpha), np.inf)
	    precision_tnc.append(prec)
	    timings_tnc.append((datetime.now() - start).total_seconds())
	callback(w0)
	out = optimize.fmin_ncg(loss, w0, fprime=gradient,
	        args=(X, y, alpha), avextol=1e-10,
	        callback=callback, maxiter=40)

	print('L-BFGS')
	timings_lbfgs = []
	precision_lbfgs = []
	w0 = np.zeros(X.shape[1])
	start = datetime.now()
	def callback(x0):
	    prec = linalg.norm(gradient(x0, X, y, alpha), np.inf)
	    precision_lbfgs.append(prec)
	    timings_lbfgs.append((datetime.now() - start).total_seconds())
	callback(w0)

	out = optimize.fmin_l_bfgs_b(loss, w0, fprime=gradient, 
	        args=(X, y, alpha), pgtol=1e-10, maxiter=200, 
	        maxfun=250, factr=1e-30, callback=None)

	# #Remember to randomize the dataset using random.randint(a,b)
	# for runs in range(RUNS):
	# 	print runs
	# 	for i in range(len(matrix)):
	# 		p=np.dot(betas,matrix[i][:801])
	# 		p=sigmoid(p)
	# 		y=matrix[i][801]
	# 		for j in range(len(matrix[i][:801])):	
	# 			x=matrix[i][j]
	# 			betas[0][j] += LAMBDA*(y-p)*x
	# 	#if runs%20 == 0:
	# #		print runs
	# 	print calculate_differential(matrix,betas)
	# print betas
	# save_to_file("logistic_regression_beta_values_sorted.bin",betas)
	# #calculate_label(matrix,betas)
main()
