#
# CISC 452
# Prediction of Blood Glusose Levels based on RTCGM Data
#
# November 10, 2016
#
# This script implements a multi-layer feed-forward neural network for glucose
# prediciton
#
# The network has 7 input nodes and 1 output node. If the current time is 'T',
# then the inputs and output represent the blood glucose measurements at the
# following times:
#   Inputs:     - T
#               - (T - 10 mins)
#               - (T - 20 mins)
#               - (T - 30 mins)
#               - (T - 40 mins)
#               - (T - 60 mins)
#               - (T - 90 mins)
#
#   Output:     - (T + 30 mins)
#

#
# TODO:
# 1. The training data should not all be loaded into memory at once in order for
#   this to be scalable. (Do batches of ~100 at a time) There are examples of
#   ways to do this online.
# 2. The network actually does a really good job of predicting from what it's
#   given. The large errors that are sometimes observed are usually caused by
#   rapid spikes or drops. Perhaps we should try a shorter time horizon (like
#   20 minutes) in order to try and reduce these errors.
# 3. Add a performance measure that evaluates the percentage of lows that are
#   succesfully identified (and counts the number of false positives).
# 4. Understand how the Adam Optimizer works.
# 5. Experiment with different node output functions. Currently, the nodes
#   simply output their activations.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt

NUM_EPOCHS = 1000 # Number of training epochs
BATCH_SIZE = 100 # Use batch optimizer (weights are only update after every 10 inputs)

# readData reads data from the specified pre-processed input data file.
# The function returns an array of input data points and an array of the
# corresponding desired outputs.
def readData(filePath) :
    x = []
    y = []
    with open(filePath, 'r') as f:
        for line in f:
            values = line.split(',')
            time1 = float(values[0])
            time2 = float(values[1])
            time3 = float(values[2])
            time4 = float(values[3])
            time5 = float(values[4])
            time6 = float(values[5])
            time7 = float(values[6])
            time8 = float(values[7])
            newPointx = [time1, time2, time3, time4, time5, time6, time7] # Input
            newPointy = [time8] # Desired Output
            x.append(newPointx)
            y.append(newPointy)
    data = [x, y]
    return data;

# evaluateNetwork runs the trained network on the the provided network and
# reports the following evaluation metrics:
#   - mean squared prediction error
#   - percentage of lows that were correctly identified
#   - percentage of highs that were corretly identified
#   - number of falsely reported lows
#   - number of falsely reported highs
#
# These metrics are defined as follows:
#   - MSE:
#       -> Average of (y_desired - y_actual)^2 for each test point
#   - Low prediction accuracy:
#       -> 100 * (Number of correct lows) / (Number of lows)
#       -> Lows are any blood glucose level less than 70 mg/dL
#   - High prediction accuracy:
#       -> 100 * (Number of correct highs) / (Number of highs)
#       -> Highs are any blood glucose level greater than 200
#   - Number of false lows:
#       -> Number of false lows where (y_desired - y_actual) > 6
#       -> Note: false alarms are not counted if the prediction error is small
#   - Number of false highs:
#       -> Number of false highs where (y_actual - y_desired) > 6
#       -> Note: false alarms are not counted if the prediciton error is small
def evaluateNetwork(session, inData, outData, x, y, y_) :
    # Compute mse:
    mse = session.run(tf.reduce_mean(tf.square(y - y_)), feed_dict={x: inData, y_: outData})
    numTestPoints = len(inData)
    numPredictedLows = 0
    numLows = 0
    numFalseLows = 0
    numPredictedHighs = 0
    numHighs = 0
    numFalseHighs = 0
    for i, inputPoint in enumerate(inData) :
        # Apply network on current point:
        predicted = session.run(y, feed_dict={x: [inputPoint]})
        desired = outData[i][0]

        # Update numLows, numHighs:
        if(desired < 70) :
            numLows += 1
        elif(desired > 200) :
            numHighs += 1

        # Update prediction counts:
        if(predicted < 70) : # If predicted low
            if(desired < 70) : # If low prediction was correct
                numPredictedLows += 1
            elif((desired - predicted) > 6) : # If low prediction was incorrect and error was 'large'
                numFalseLows += 1
        elif(predicted > 200) : # If predicted high
            if(desired > 200) : # If high prediction was correct
                numPredictedHighs += 1
            elif((predicted - desired) > 6) : # If high prediction was incorrect and error was 'large'
                numFalseHighs += 1

    # Print results:
    print('Number of test points: ', numTestPoints)
    print('Number of lows: ', numLows)
    print('Number of highs: ', numHighs)
    print("Number of 'normal' points: ", numTestPoints - numLows - numHighs)
    print('') # New line
    print('MSE: ', mse)
    print('')
    print('Low prediction accuracy: ', 100 * numPredictedLows / numLows, '%')
    print('Number of false lows: ', numFalseLows)
    print('')
    print('High prediction accuracy: ', 100 * numPredictedHighs / numHighs, '%')
    print('Number of false highs: ', numFalseHighs)
# End evaluateNetwork(...)

def main(_):
    # Import the training data and test data:
    trainData_in, trainData_out = readData('tblADataRTCGM_Blind_Baseline_Split_output/1_train.csv')
    testData_in, testData_out = readData('tblADataRTCGM_Blind_Baseline_Split_output/1_test.csv')

    ### Create the model ###
    x = tf.placeholder(tf.float32, [None, 7]) # Input placeholder

    # Weights from inputs to first hidden layer (15 nodes):
    Wh1 = tf.Variable(tf.random_uniform([7, 15], minval = -1, maxval = 1, dtype = tf.float32))
    # Bias for first hidden layer:
    bh1 = tf.Variable(tf.zeros([1, 15]))

    # Weights from first hidden layer to second (15 nodes):
    Wh2 = tf.Variable(tf.random_uniform([15, 15], minval = -1, maxval = 1, dtype = tf.float32)) # The weights from each of the 784 inputs to the 10 output nodes
    # Bias for second hidden layer:
    bh2 = tf.Variable(tf.zeros([1, 15])) # One bias input for each of the 10 output nodes

    # Weights from second hidden layer to output layer (1 node):
    Wo = tf.Variable(tf.random_uniform([15, 1], minval = -1, maxval = 1, dtype = tf.float32))
    # Bias to output node:
    bo = tf.Variable(tf.zeros([1, 1]))

    # Nodes have no output function (they simply output their activation):
    h1 = tf.add(tf.matmul(x, Wh1), bh1) # Hidden layer 1 output
    h2 = tf.add(tf.matmul(h1, Wh2), bh2) # Hidden layer 2 output
    y = tf.add(tf.matmul(h2, Wo), bo) # Network output

    y_ = tf.placeholder(tf.float32, [None, 1]) # Desired output placeholder

    # Error function to be minimized is the mean square error:
    loss = tf.reduce_mean(tf.square(y - y_))

    # Define training algorithm (Adam Optimizer):
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

    ### Train ###
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    errors = []
    for i in range(NUM_EPOCHS): # 1000 training epochs
        # Train with one batch at a time (This was tested and found to be less effective than):
        #for start, end in zip(range(0, len(trainData_in), BATCH_SIZE), range(BATCH_SIZE, len(trainData_in), BATCH_SIZE)):
        #    sess.run(train_step, feed_dict={x: trainData_in[start:end], y_: trainData_out[start:end]})
        sess.run(train_step, feed_dict={x: trainData_in, y_: trainData_out})
        mse = sess.run(tf.reduce_mean(tf.square(y - y_)), feed_dict={x: testData_in, y_: testData_out})
        errors.append(mse)
        #print(mse)
    ### Test ###
    # Apply network on all test data and compute MSE:
    #mse = sess.run(tf.reduce_mean(tf.square(y - y_)), feed_dict={x: testData_in, y_: testData_out})
    #print('MSE = ', mse)
    evaluateNetwork(sess, testData_in, testData_out, x, y, y_)
    # Output the desired and actual outputs for each test data point
    #for i, inputPoint in enumerate(testData_in) :
    #    output = sess.run(y, feed_dict={x: [inputPoint]})
    #    print('desired: ', testData_out[i], ', actual: ', output)

    # Plot the MSE throughout training
    #plt.plot(errors)
    #plt.xlabel('#epochs')
    #plt.ylabel('MSE')
    #plt.show()

    # Uncomment this to compute the MSE for the current network on a different patient:
    #testData_in, testData_out = readData('tblADataRTCGM_Blind_Baseline_Split_output/78_test.csv')
    #mse = sess.run(tf.reduce_mean(tf.square(y - y_)), feed_dict={x: testData_in, y_: testData_out})
    #print('MSE2 = ', mse)


if __name__ == '__main__':
    tf.app.run()
