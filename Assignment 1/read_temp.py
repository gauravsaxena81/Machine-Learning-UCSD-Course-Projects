import numpy as np

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

f=open('/Users/suvir/Documents/Logistic Regression/temp')
l=f.readlines()
f.close()
array=[]
for line in l:
	for element in line.strip().split():
		array.append(float(element))
#print array

a=np.array((1,801))
a=np.copy(array)
print a

save_to_file("logistic_regression_beta_values.bin",a)
b=load_from_file("logistic_regression_beta_values.bin")
print np.array_equal(a,b)