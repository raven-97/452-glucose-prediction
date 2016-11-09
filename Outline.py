################################ 452 Project ################################
#
#
# Input is a .csv file that has 0, -10, -20, -30, -40, -60, -90,
# Intention is to accurately predict +30 (all times measured in minutes)
# Current time is always from around 1AM to 7AM
# So, inputs are taken from around 11:30PM onward (when patient is sleeping)
# We ill be given test, validation, and training data for about 15 days/patient
#
#
# So, we want to use TensorFlow to implement a multilayer network with:
#     - 7 inputs (shown above)
#     - Varied size of hidden layer
#     - 1 output (prediction of real value)
#     - Error Correction
#
#
# Demonstration is training it on a new patient and then comparing a run on the
# "test" data to a new patient. This should show that it clearly performs better
# on the first patient, since it has been trained on them.
#
#


import tensorflow as tf
import numpy as np


def get_data():
	A=[]
	B=[]

	with open ("/home/group/452-project/tblADataRTCGM_Blind_Baseline_Split_output/1.csv", "r") as file:
		for line in file:
			B = line.split(",")
			A.append(B)
			
			list_element = A[line]					# access current list element
			last = list_element[-1]
			last = last.substring(0,last.length()-4)
			
			array = range(0,len(list_element))
			for i in array:
				list_element[i] = float(list_element[i])			
		file.closed

	
	return A
	
def main():
	
	test_data = get_data()
	print test_data[1]
	
	
	
	
	
	
	
if __name__ == '__main__':
	main()
	
	
	
	
	
	
	
	
	
	
	
	
	
	