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












