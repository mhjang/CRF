__author__ = 'mhjang'

import numpy as np
import itertools
import math
from scipy import optimize


charMap = {0:'e', 1:'t', 2:'a', 3:'i', 4:'n', 5:'o', 6:'s', 7:'h', 8:'r', 9:'d'}
invCharMap = {'e':0, 't':1, 'a':2, 'i':3, 'n':4, 'o':5, 's':6, 'h':7, 'r':8, 'd':9}

N = 50

f = open('data/train_words.txt', 'r')
trainWords = f.readlines()
word_feature = list()
for i in range(1, (N+1)):
    filename = 'data/train_img' + str(i) + ".txt"
    word = trainWords[i-1][:-1]
    feature = np.genfromtxt(filename)
    word_feature.append((word, feature))



# for problem 2.1
def cliquePotential(features, featureParams, transitiveParams):
    psimaps = [np.zeros((10,10)) for i in range(len(features)-1)]
    # for cluster 1
    # i: index of y0
    # j: index of y1
    for k in range(len(features)-1):
        for i in range(10):
            for j in range(10):
                if k != len(features)-2:
                    psimaps[k][i][j] = sum(featureParams[i] * features[k]) + transitiveParams[i][j]
                else:
                    psimaps[k][i][j] = sum(featureParams[i] * features[k]) + sum(featureParams[j] * features[k+1]) + transitiveParams[i][j]
  #  printETRtable(psimaps[-1])
    return psimaps



# for problem 2.2
def computeMessagePassing(features, potentialMap):
#    potentialMap = cliquePotential(features)
    cliqueNumber = len(potentialMap)
    forwardMessages = [np.zeros(10) for i in range((cliqueNumber-1))]
    backwardMessages = [np.zeros(10) for i in range((cliqueNumber-1))]

    potentialMapExp = np.exp(potentialMap)
    # very first forward message
    forwardMessages[0] = np.log(np.sum(potentialMapExp[0], axis=0))

    # rest of the chains
    for k in range(1, len(forwardMessages)):
        forwardMessages[k] = np.log(np.sum(potentialMapExp[k], axis=0) + forwardMessages[k-1])

   # very first backward message
    for i in range(10):
        sum = 0.0
        for j in range(10):
            sum = sum + math.exp(potentialMap[cliqueNumber-1][i][j])

        backwardMessages[cliqueNumber-2][i] = math.log(sum)

   # rest of the chains
    for k in range(cliqueNumber-3,-1,-1):
     for i in range(10):
        sum = 0.0
        for j in range(10):
            sum = sum + math.exp(potentialMap[k+1][i][j] + backwardMessages[k+1][j])
        backwardMessages[k][i] = math.log(sum)

    return [forwardMessages, backwardMessages]


# for problem 2.3
def computeLogBeliefs(feature, potentialMap, featureParams, transitiveParams):
 #   potentialMap = cliquePotential(feature)
    messages = computeMessagePassing(feature, potentialMap)
    forwardMessages = messages[0]
    backwardMessages = messages[1]
    cliqueNumber = len(potentialMap)

    beliefs = [np.zeros((10,10)) for i in range((cliqueNumber))]

    beliefs[0] = potentialMap[0] + backwardMessages[0]
    for k in range(1, cliqueNumber-1):
        beliefs[k] = potentialMap[k] + forwardMessages[k-1] + backwardMessages[k]

    beliefs[cliqueNumber-1] =  potentialMap[cliqueNumber-1] + forwardMessages[cliqueNumber-2]
    return beliefs


# for problem 2.5
def predictStringWithMarginals(feature):
    beliefs = computeLogBeliefs(feature)
    expBeliefs = [np.exp(belief) for belief in beliefs]
    marginal_probs = [belief / np.sum(belief, axis=None) for belief in expBeliefs]

    predictedWord = [charMap[np.argmax(np.max(marginals, axis = 1))] for marginals in marginal_probs]
    predictedWord.append(charMap[np.argmax(np.max(marginal_probs[-1], axis = 0))])
 #   print(''.join(predictedWord))
    return ''.join(predictedWord)


def derivativeFunctions(x):
    featureParams = np.reshape(x[:10*321], (10, 321))
    transitiveParams = np.reshape(x[10*321:], (10, 10))
    N = 50
    result = np.zeros(10*321 + 10*10)
    f = gradientFunctionFeatureParam(featureParams, transitiveParams, N)
    t = gradientFunctionTransitiveParam(featureParams, transitiveParams, N)
    flatten_f = np.reshape(f, (10*321))
    flatten_t = np.reshape(t, (10*10))
    result = np.append(flatten_f, flatten_t)
    return result

# for assignment 2(B)
def computeLogLikelihood(x):
    featureParams = np.reshape(x[:10*321], (10, 321))
    transitiveParams = np.reshape(x[10*321:], (10, 10))
    sum = 0
    for i in range(N):
        word = word_feature[i][0]
        feature = word_feature[i][1]
        potentialMap = cliquePotential(feature, featureParams, transitiveParams)
        negEnergyWord = getCliquePotentialValue(feature, word, potentialMap, featureParams, transitiveParams)
        beliefs = computeLogBeliefs(feature, potentialMap, featureParams, transitiveParams)
        maxValue = np.max(beliefs[0], axis=None)
        partition = maxValue + np.log(np.sum(np.exp(beliefs[0]-maxValue), axis=None))
        sum += negEnergyWord - partition
  #      print("negative energy: " + str(negEnergyWord) + ", partition:" + str(partition))
    print(str(sum/N))
    return sum/N*(-1)

def gradientFunctionTransitiveParam(featureParams, transitiveParams, N):
    sum = 0
    gradient = np.zeros((10, 10))
    for i in range(N):
        word = word_feature[i][0]
        feature = word_feature[i][1]
        potentialMap = cliquePotential(feature, featureParams, transitiveParams)
        negEnergyWord = getCliquePotentialValue(feature, word, potentialMap, featureParams, transitiveParams)

        beliefs = computeLogBeliefs(feature, potentialMap, featureParams, transitiveParams)
        expBeliefs = [np.exp(belief) for belief in beliefs]
        marginal_probs = [belief / np.sum(belief, axis=None) for belief in expBeliefs]

        for j in range(len(word)-1):
            binary = np.zeros((10, 10))
            binary[invCharMap[word[j]]][invCharMap[word[j+1]]] = 1
            gradient += binary - marginal_probs[j]

    return gradient/N*(-1)


def gradientFunctionFeatureParam(featureParams, transitiveParams, N):
    sum = 0
    gradient = np.zeros((10, 321))
    for i in range(N):
        word = word_feature[i][0]
        feature = word_feature[i][1]
        potentialMap = cliquePotential(feature, featureParams, transitiveParams)
        negEnergyWord = getCliquePotentialValue(feature, word, potentialMap, featureParams, transitiveParams)

        beliefs = computeLogBeliefs(feature, potentialMap, featureParams, transitiveParams)
        expBeliefs = [np.exp(belief) for belief in beliefs]
        marginal_probs = [belief / np.sum(belief, axis=None) for belief in expBeliefs]
        for j in range(len(word)):
            for k in range(len(charMap.keys())):
                if word[j] == charMap[k]:
                    binaryOccurrence = 1
                else:
                    binaryOccurrence = 0
                if j == 0:
                    gradient[k] += (binaryOccurrence - np.sum(marginal_probs[0], axis=1)[k]) * feature[j]
                else:
                    gradient[k] += (binaryOccurrence - np.sum(marginal_probs[j-1], axis=0)[k]) * feature[j]


      #  print(str(sum/(N-1)))
    return gradient/N*(-1)



def getCliquePotentialValue(feature, word, potentialMap, featureParams, transitiveParams):
     return np.sum([potentialMap[i][invCharMap[word[i]]][invCharMap[word[i+1]]] for i in range(len(word)-1)])
def main():

 #   featureParams = np.genfromtxt('model/feature-params.txt')
 #   transitiveParams = np.genfromtxt('model/transition-params.txt')



#    gradientFunctionFeatureParam(featureParams, transitiveParams)
 #   gradientFunctionTransitiveParam(featureParams, transitiveParams)

    x0 = np.zeros(10*321 + 10*10)
    optimize.fmin_cg(computeLogLikelihood, x0, fprime=derivativeFunctions, args=())
    print(x0)

 #   computeLogLikelihood(featureParams, transitiveParams)
#    f = open('data/test_words.txt', 'r')
#    testWords = f.readlines()
#    accuracy = np.zeros(200);
#    correct = 0
#    incorrect =0
#    for i in range(1,201):
#        filename = 'data/test_img' + str(i) + ".txt"
#        features = np.genfromtxt(filename)
#        word = (str)(predictStringWithMarginals(features))
#        trueWord = testWords[i-1]
#        bitDifference = 0
#        for k in range(len(word)):
#            if word[k] == trueWord[k]:
#                correct += 1
#                bitDifference +=1
#            else:
#                incorrect += 1
#        accuracy[i-1] = correct/len(word)
 #       print(word + "\t" + trueWord + "\t" + (str)(bitDifference))
#    print(str(np.sum(accuracy)/200))
#    print((str)((correct)/(correct + incorrect)))


if __name__ == "__main__":
    main()