__author__ = 'Myung-ha Jang'

import numpy as np
import itertools
import math

# character index
e = 0
t = 1
a = 2
i = 3
n = 4
o = 5
s = 6
h = 7
r = 8
d = 9

charMap = {0:'e', 1:'t', 2:'a', 3:'i', 4:'n', 5:'o', 6:'s', 7:'h', 8:'r', 9:'d'}
invCharMap = {'e':0, 't':1, 'a':2, 'i':3, 'n':4, 'o':5, 's':6, 'h':7, 'r':8, 'd':9}

featureParams = np.genfromtxt('model/feature-params.txt')
transitiveParams = np.genfromtxt('model/transition-params.txt')

# for problem 2.1
def cliquePotential():
    features = np.genfromtxt('data/test_img1.txt')
    psi_y1_y2_map = np.zeros((10,10))
    psi_y2_y3_map = np.zeros((10,10))
    psi_y3_y4_map = np.zeros((10,10))
    # for cluster 1
    # i: index of y0
    # j: index of y1
    print("Cluster Potential #1")
    for i in range(10):
        psi_y1 = sum(featureParams[i] * features[0])
        for j in range(10):
            psi_y1_y2 = transitiveParams[i][j]
    #       uncomment this line to print the entire table
    #        print((str)(psi_y1 + psi_y1_y2), end="\t")
            psi_y1_y2_map[i][j] = psi_y1 + psi_y1_y2
    #    print()

    printETRtable(psi_y1_y2_map)

    print("Cluster Potential #2")
    for i in range(10):
        psi_y2 = sum(featureParams[i] * features[1])
        for j in range(10):
            psi_y2_y3 = transitiveParams[i][j]
    #       uncomment this line to print the entire table
    #        print((str)(psi_y2 + psi_y2_y3), end="\t")
            psi_y2_y3_map[i][j] = psi_y2 + psi_y2_y3
    #    print()
    printETRtable(psi_y2_y3_map)

    print("Cluster Potential #3")
    for i in range(10):
        psi_y3 = sum(featureParams[i] * features[2])
        for j in range(10):
            psi_y4 = sum(featureParams[j] * features[3])
            psi_y3_y4 = transitiveParams[i][j]
    #       uncomment this line to print the entire table
    #        print((str)(psi_y3 + psi_y4 + psi_y3_y4), end="\t")
            psi_y3_y4_map[i][j] = psi_y3 + psi_y4 + psi_y3_y4
   #     print()
    printETRtable(psi_y3_y4_map)

    return [psi_y1_y2_map, psi_y2_y3_map, psi_y3_y4_map]

# The assignment asks to print 3 x 3 block of entires between the labels 'e','t','r'
# This simply takes an array to print that area.
def printETRtable(map):
    print(str(map[e][e]) + "\t" + str(map[e][t]) + "\t" + str(map[e][r]))
    print(str(map[t][e]) + "\t" + str(map[t][t]) + "\t" + str(map[t][r]))
    print(str(map[r][e]) + "\t" + str(map[r][t]) + "\t" + str(map[r][r]))

def printETtable(map):
    print(str(map[e][e]) + "\t" + str(map[e][t]))
    print(str(map[t][e]) + "\t" + str(map[t][t]))



# for problem 2.2
def computeMessagePassing():
    potentialMap = cliquePotential()
    psi_y1_y2 = potentialMap[0]
    psi_y2_y3 = potentialMap[1]
    psi_y3_y4 = potentialMap[2]

    delta_1_2 = np.zeros(10)
    delta_3_2 = np.zeros(10)
    delta_2_3 = np.zeros(10)
    delta_2_1 = np.zeros(10)

    deltaMap = list()

    # delta_{1->2}(Y2) = log \sum_{y1}(exp(Y1, Y2)
    print("Delta 1->2")
    for i in range(10):
        sum = 0.0
        for j in range(10):
            sum = sum + math.exp(psi_y1_y2[j][i])
        print(charMap[i] + "\t" + str(math.log(sum)))
        delta_1_2[i] = math.log(sum)

    print("Delta 2->3")
    # delta_{2->3}(Y3) = log \sum_{y2}(exp(Y3, Y4)
    for i in range(10):
        sum = 0.0
        for j in range(10):
            sum = sum + math.exp(psi_y2_y3[j][i] + delta_1_2[j])
        print(charMap[i] + "\t" + str(math.log(sum)))
        delta_2_3[i] = math.log(sum)

    print("Delta 3->2")

    # delta_{3->2}(Y3) = log \sum_{y4}(exp(Y3, Y4)
    for i in range(10):
        sum = 0.0
        for j in range(10):
            sum = sum + math.exp(psi_y3_y4[i][j])
        print(charMap[i] + "\t" + str(math.log(sum)))

        delta_3_2[i] = math.log(sum)

    print("Delta 2->1")
    # delta_{2->1}(Y3) = log \sum_{y2}(exp(Y2, Y3)
    for i in range(10):
        sum = 0.0
        for j in range(10):
            sum = sum + math.exp(psi_y2_y3[i][j] + delta_3_2[j])
        print(charMap[i] + "\t" + str(math.log(sum)))
        delta_2_1[i] = math.log(sum)

    return [delta_1_2, delta_2_3, delta_3_2, delta_2_1]


# for problem 2.3
def computeLogBeliefs():
    potentialMap = cliquePotential()
    psi_y1_y2 = potentialMap[0]
    psi_y2_y3 = potentialMap[1]
    psi_y3_y4 = potentialMap[2]

    deltaMap = computeMessagePassing()
    delta_1_2 = deltaMap[0]
    delta_2_3 = deltaMap[1]
    delta_3_2 = deltaMap[2]
    delta_2_1 = deltaMap[3]

    belief_1_2 = np.zeros((10,10))
    belief_2_3 = np.zeros((10,10))
    belief_3_4 = np.zeros((10,10))

    # delta_{1->2}(Y2) = log \sum_{y1}(exp(Y1, Y2)
    print("B(Y1, Y2)")
    for i in range(10):
        sum = 0.0
        for j in range(10):
            belief_1_2[i][j] = (psi_y1_y2[i][j]) + delta_2_1[j]
    #    print(charMap[i] + "\t" + str(belief_1_2[i]))
    printETtable(belief_1_2)

    print("B(Y2, Y3)")
    # delta_{2->3}(Y3) = log \sum_{y2}(exp(Y3, Y4)
    for i in range(10):
        sum = 0.0
        for j in range(10):
            belief_2_3[i][j] = psi_y2_y3[i][j] + delta_1_2[i] + delta_3_2[j]
    #    print(charMap[i] + "\t" + str(belief_2_3[i]))
    printETtable(belief_2_3)

    print("B(Y3, Y4)")
    # delta_{3->2}(Y3) = log \sum_{y4}(exp(Y3, Y4)
    for i in range(10):
        sum = 0.0
        for j in range(10):
            belief_3_4[i][j] =  psi_y3_y4[i][j] + delta_2_3[i]
#        print(charMap[i] + "\t" + str(belief_3_4[i]))
    printETtable(belief_3_4)
    return [belief_1_2, belief_2_3, belief_3_4]


# for problem 2.4
def computeMarginalProbDistribution():
    belief_map = computeLogBeliefs()
    belief_1_2 = np.exp(belief_map[0])
    belief_2_3 = np.exp(belief_map[1])
    belief_3_4 = np.exp(belief_map[2])

    Z = np.sum(belief_1_2, axis=None)
    print("P(y11, y12 | x1)")
    marginal_prob_1_2 = belief_1_2 / Z

   # print(marginal_1)

    printETRtable(marginal_prob_1_2)

    print("P(y12, y13 | x1)")
    Z = np.sum(belief_2_3, axis= None)
    marginal_prob_2_3 = belief_2_3 / Z
    printETRtable(marginal_prob_2_3)

    print("P(y13, y14 | x1)")
    Z = np.sum(belief_3_4, axis=None)
    marginal_prob_3_4 = belief_3_4 / Z
    printETRtable(marginal_prob_3_4)


    print("Marginal probability for the 1st position")
    marginal = np.sum(marginal_prob_1_2, axis = 1)
    for i in range(10):
        print(charMap[i] + "\t" + str(marginal[i]))


    print("Marginal probability for the 2nd position")
    marginal = np.sum(marginal_prob_1_2, axis = 0)
    for i in range(10):
        print(charMap[i] + "\t" + str(marginal[i]))

    print("Marginal probability for the 3rd position")
    marginal = np.sum(marginal_prob_2_3, axis = 0)
    for i in range(10):
        print(charMap[i] + "\t" + str(marginal[i]))

    print("Marginal probability for the 4th position")
    marginal = np.sum(marginal_prob_3_4, axis = 0)
    for i in range(10):
        print(charMap[i] + "\t" + str(marginal[i]))


    return [marginal_prob_1_2, marginal_prob_2_3, marginal_prob_3_4]

# for problem 2.5
def predictMostProbableString():
    marginalMap = computeMarginalProbDistribution()
    marginal_prob_1_2 = marginalMap[0]
    marginal_prob_2_3 = marginalMap[1]
    marginal_prob_3_4 = marginalMap[2]

 #   print(marginal_prob_1_2)
    # take only max values for each row
    maxmarginal = np.max(marginal_prob_1_2, axis = 1)
    print(str(np.argmax(maxmarginal)))
    ch = charMap[np.argmax(maxmarginal)]
    print(ch, end = "")

    maxmarginal = np.max(marginal_prob_2_3, axis = 1)
    ch = charMap[np.argmax(maxmarginal)]
    print(ch, end = "")


    maxmarginal = np.max(marginal_prob_3_4, axis = 1)
    ch = charMap[np.argmax(maxmarginal)]
    print(ch, end = "")

    maxmarginal = np.max(marginal_prob_3_4, axis = 0)
    ch = charMap[np.argmax(maxmarginal)]
    print(ch, end = "")


def printNodePotentialTable():
    img1Features = np.genfromtxt('data/test_img1.txt')

    for i in range(10):
        print(charMap[i], end = "\t")
        for j in range(4):
            print((str)(sum(img1Features[j] * featureParams[i])), end = "\t")
        print()

#def computeNodePotential(word, features, i):
#    return sum(featureParams[invCharMap[word[i]]] * features[i])


def computeNodePotential(word, features):
    potentialSum = 0
    for i in range(len(features)):
      potentialSum += sum(featureParams[invCharMap[word[i]]] * features[i])
    return potentialSum

def computeTransitiveProbability(word, i):
    return transitiveParams[invCharMap[word[i]]][invCharMap[word[i+1]]]


def computeTransitiveProbability(word):
    sum = 0
    for i in range(len(word)-1):
        sum += transitiveParams[invCharMap[word[i]]][invCharMap[word[i+1]]]
    return sum

def prob2():
    features = np.genfromtxt('data/test_img1.txt')
    print("tree:" + str(computeNodePotential('tree', features) + computeTransitiveProbability('tree')))

    features = np.genfromtxt('data/test_img2.txt')
    print("net:" + str(computeNodePotential('net', features) + computeTransitiveProbability('net')))

    features = np.genfromtxt('data/test_img3.txt')
    print("trend:" + str(computeNodePotential('trend', features) + computeTransitiveProbability('trend')))


def exhaustiveSummation(featureFile):
    feature = np.genfromtxt(featureFile)
    possibleSeq = itertools.product(invCharMap.keys(), repeat = len(feature))

    exSum = 0.0
    wordSum = -10000
    maxSum = 0.0
    likelyLabel = ""
    for seq in possibleSeq:
        word = ''.join(seq)
        wordSum = computeNodePotential(word, feature) + computeTransitiveProbability(word)
        wordSum = math.exp(wordSum)
        if wordSum > maxSum:
            likelyLabel = word
            maxSum = wordSum
        exSum = exSum + wordSum

    print("log partition:" + str(math.log(exSum)))
    print(str(maxSum/exSum) + ", " + likelyLabel)

def marginalProb(word):
    features = np.genfromtxt('data/test_img1.txt')
    for k in range(len(word)):
        marginalSum = 0.0
        for i in range(10):
            marginalSum = marginalSum + sum(featureParams[i] * features[k])
        prob = sum(featureParams[invCharMap[word[k]]] * features[k])
        print(word[k] + ":" + str(math.exp(prob)/math.exp(marginalSum)))



def main():

######## Prob 2.1 ##########
#     cliquePotential()

######## Prob 2.2 ##########
#     computeMessagePassing()

######## Prob 2.3 ##########
#        computeLogBeliefs()

######## Prob 2.4 #########
    computeMarginalProbDistribution()

######## Prob 2.5 #########
#    predictMostProbableString()
#    computeTransitiveProbability('tree')


#    features = np.genfromtxt('data/test_img1.txt')




####### Prob 1.1 ##########
#    printNodePotentialTable()
#    computeNodePotential('tree', features)


####### Prob 1.2 ###########
#   prob2()


####### Prob 1.3 & 1.4 #########
# exhaustiveSummation('data/test_img1.txt')
 #   exhaustiveSummation('data/test_img2.txt')
 #   exhaustiveSummation('data/test_img3.txt')

 ####### PProb 1.5 #######
 #    marginalProb('tree')

if __name__ == "__main__":
    main()