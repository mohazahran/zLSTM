'''
Created on Feb 21, 2018

@author: mohame11
'''
import numpy as np
import nltk
import itertools
import math
import random
from scipy import special
import pickle as pkl
import re
from enum import Enum

class DATA_GENERATION (Enum):
    SAMPLING = 1
    MAX_LIKELIHOOD = 2
   
    
def sigmoid(x):
    #return 1. / (1. + np.exp(-x))
    return special.expit(x)

    
def clean_Data(myData):
    cleanedData = []   
    p1 = re.compile("\w+-\w+")   
    p2 = re.compile("\W\d+[A-Za-z]+")
    p3 = re.compile("[A-Za-z]+\d+\W")
    p4 = re.compile("[^A-Za-z0-9\s']") #remove special chars
    p5 = re.compile("[^A-Za-z](\d+\s*)+") #consectutive NUMs
    for line in myData:    
        line = line.strip()    
        matches = p1.findall(line)
        for m in matches:
            line = line.replace(m, m.replace('-',''))
        matches = p2.findall(line)
        for m in matches:
            id = 1
            for c in m:
                if c.isdigit():
                    id +=1
                elif(id != 1):
                    break
            s = m[:id] + ' ' + m[id:].strip()
            line = line.replace(m, s)
        
        matches = p3.findall(line)
        for m in matches:
            id = 0
            for c in m:
                if c.isdigit():
                    break
                else:
                    id +=1
            s = m[:id] + ' ' + m[id:].strip()
            line = line.replace(m, s)
        
        matches = p4.findall(line)
        for m in matches:
            line = line.replace(m, m.replace(m,' '))
            
        matches = p5.findall(line)
        for m in matches:
            line = line.replace(m, ' num ')
            
        #line = line.replace('unk', 'UNK')
            
        cleaned = ' '.join(line.split())          
        cleaned = cleaned.lower()
        cleaned = cleaned.strip()
        if len(cleaned) <= 1:
            continue
        cleanedData.append(cleaned)        
    return cleanedData


def preProcessing_charBased(filePath='',  v2i = None, i2v = None):
    f = open(filePath, 'r') 
    startToken = '['
    endToken = ']'
    if v2i == None or i2v == None:
        chars = f.read()
        uniqueChars = set(chars.replace('\n', ' '))
        uniqueChars.add(startToken); uniqueChars.add(endToken)
        v2i = { ch:i for i,ch in enumerate(uniqueChars) }
        i2v = { i:ch for i,ch in enumerate(uniqueChars) }
    
        print 'The number of chars = %d' % len(chars)
        print "The number of unique chars = %d." % len(v2i)
    
        sentences = chars.split('\n')
    else:
        sentences = f.read().split('\n')
        
    sentences = ["%s %s %s" % (startToken, x, endToken) for x in sentences]
    X,Y = [],[]
    for sent in sentences:
        X.append(np.asarray( [v2i[c] for c in sent[:-1]] ))
        Y.append(np.asarray( [v2i[c] for c in sent[1:]]))
        
    
    return X, Y, v2i, i2v


def preProcessing(filePath = '', vocabSize = 50, v2i = None, i2v = None):
    vocabulary_size = vocabSize
    unknown_token = "UNK"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading file..."
    f = open(filePath, 'r')
    # Split full comments into sentences
    sentences = f.readlines()
    #sentences = clean_Data(sentences)
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))
         
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent.replace('unk','UNK')) for sent in sentences]
    
    
    if v2i == None or i2v == None:
        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print "Found %d unique words tokens." % len(word_freq.items())
         
        # Get the most common words and build index_to_word and word_to_index vectors
        vocab = word_freq.most_common(vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
         
        print "Using vocabulary size %d." % vocabulary_size
        print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
    else:
        word_to_index = v2i; index_to_word = i2v
     
    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
        #dropout all unknown words
        #tokenized_sentences[i] = [w for w in sent if w in word_to_index]
     
    #print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
     
    # Create the training data
    X = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    Y = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    return X, Y, word_to_index, index_to_word


def generateText(lstm = None, v2i = None, i2v = None, startSeed = '[', genLength = 100, instanceCount = 20, dataGenerationType = DATA_GENERATION.MAX_LIKELIHOOD):
    if startSeed is None: 
        startSeed = random.choice(v2i.keys())
    if startSeed not in v2i:
        print 'Word is not in vocab, try different word!'
        return
    generatedSamples = []
    for inst in range(instanceCount):
        inst = startSeed
        genValue = startSeed
        ht_1 = np.zeros((lstm.hiddenDim,1))
        ct_1 = np.zeros((lstm.hiddenDim,1))
        for t in range(genLength):
            xt = np.zeros((lstm.inputDim,1))
            xt[v2i[genValue]][0] = 1.0
            ht, ct = lstm.generate(xt, ht_1, ct_1)
            gt = np.dot(lstm.weights['Wout'], ht) + lstm.weights['bout']
            softmaxPredictions = lstm.stable_softmax(gt)
            
            if dataGenerationType == DATA_GENERATION.MAX_LIKELIHOOD:
                genId = softmaxPredictions.T[0].argmax()
            else:
                genId = np.random.choice(range(lstm.inputDim), p=softmaxPredictions.T[0].ravel())
            #print sum(softmaxPredictions[0])
            #genWordId = np.random.choice(range(lstm.inputDim), p=softmaxPredictions.T[0].ravel())
            genValue = i2v[genId]
            inst += genValue
        
            ht_1 = ht
            ct_1 = ct
        generatedSamples.append(inst)
    return generatedSamples
    

def simulateData():
    lst = pkl.load(open('lstm.pkl', 'rb'))
    lstm = lst[0]
    v2i = lst[1]
    i2v = lst[2]
    sent = generateText(lstm, v2i, i2v, None, 1310)
    print sent
    

class zLSTM(object):
    '''
    An implementation of LSTM class trained by backpropagation through time
    '''
    def __init__(self, inputDim=10, hiddenDim=10, learningRate = 0.01, clipGradients = False, useAdaGrad = False, batchSize = 1, BPTT_length = 25):
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.clipGradients = clipGradients
        self.useAdaGrad = useAdaGrad
        self.batchSize = batchSize
        self.BPTT_length = BPTT_length
        
        #for experiments with evolving the states of C
        self.epoch = 0
        
        #self.Wz = np.random.uniform(-np.sqrt(1./inputDim), np.sqrt(1./inputDim), (hiddenDim, inputDim))
        Wz = np.random.uniform(-0.1, 0.1, (hiddenDim, inputDim))
        Wi = np.random.uniform(-0.1, 0.1, (hiddenDim, inputDim))
        Wf = np.random.uniform(-0.1, 0.1, (hiddenDim, inputDim))
        Wo = np.random.uniform(-0.1, 0.1, (hiddenDim, inputDim))
        
        Rz = np.random.uniform(-0.1, 0.1, (hiddenDim, hiddenDim))
        Ri = np.random.uniform(-0.1, 0.1, (hiddenDim, hiddenDim))
        Rf = np.random.uniform(-0.1, 0.1, (hiddenDim, hiddenDim))
        Ro = np.random.uniform(-0.1, 0.1, (hiddenDim, hiddenDim))
        
        bz = np.random.uniform(-0.1, 0.1, (hiddenDim,1))
        bi = np.random.uniform(-0.1, 0.1, (hiddenDim,1))
        #self.bf = np.ones((hiddenDim,1)) #better to be initialized to all ones.
        bf = np.random.uniform(-0.1, 0.1, (hiddenDim,1)) 
        bo = np.random.uniform(-0.1, 0.1, (hiddenDim,1))
        
        Wout = np.random.uniform(-0.1, 0.1, (inputDim, hiddenDim))
        bout = np.random.uniform(-0.1, 0.1, (inputDim,1))
        
        self.weights = {}
        self.weights['Wz'], self.weights['Wi'], self.weights['Wf'], self.weights['Wo'] = Wz, Wi, Wf, Wo
        self.weights['Rz'], self.weights['Ri'], self.weights['Rf'], self.weights['Ro'] = Rz, Ri, Rf, Ro
        self.weights['bz'], self.weights['bi'], self.weights['bf'], self.weights['bo'] = bz, bi, bf, bo 
        self.weights['Wout'] = Wout
        self.weights['bout'] = bout
        
        #memory for gradients for adagrad
        mdWz = np.zeros(Wz.shape) ; mdWi = np.zeros(Wi.shape); mdWf = np.zeros(Wf.shape); mdWo = np.zeros(Wo.shape)
        mdRz = np.zeros(Rz.shape); mdRi = np.zeros(Ri.shape); mdRf = np.zeros(Rf.shape); mdRo = np.zeros(Ro.shape)
        mdbz = np.zeros(bz.shape); mdbi = np.zeros(bi.shape); mdbf = np.zeros(bf.shape); mdbo = np.zeros(bo.shape)
        mdWout = np.zeros(Wout.shape)
        mdbout = np.zeros(bout.shape)
        
        self.gradientMemory={}
        self.gradientMemory['Wz'] = mdWz; self.gradientMemory['Wi'] = mdWi; self.gradientMemory['Wf'] = mdWf; self.gradientMemory['Wo'] = mdWo
        self.gradientMemory['Rz'] = mdRz; self.gradientMemory['Ri'] = mdRi; self.gradientMemory['Rf'] = mdRf; self.gradientMemory['Ro'] = mdRo
        self.gradientMemory['bz'] = mdbz; self.gradientMemory['bi'] = mdbi; self.gradientMemory['bf'] = mdbf; self.gradientMemory['bo'] = mdbo; 
        self.gradientMemory['Wout'] = mdWout
        self.gradientMemory['bout'] = mdbout
        
        
    def forwardPass(self, inputSeq, Ht_1 = None, Ct_1 = None): #inputSeq is a sequence of input vectors (e.g. an input sentence)
        inputCount = len(inputSeq) #e.g. number of words in a sentence
        
        # The outputs at each time step.
        G = {}
        H = {}
        O = {}
        C = {}
        F = {}
        I = {}
        Z = {}
        softmaxPredictions = {}
        
        if Ht_1 is None: H[-1] = np.zeros((self.hiddenDim,1)) # initial hidden state
        else: H[-1] = Ht_1
        
        if Ct_1 is None: C[-1] = np.zeros((self.hiddenDim,1)) # initial hidden state
        else: C[-1] = Ct_1
        
        for t in range(inputCount): # for each item in a sequence (e.g. word in a sentence)
            xt = np.zeros((self.inputDim,1))
            xt[inputSeq[t]][0] = 1.0
            
            zt_ = np.dot(self.weights['Wz'], xt) + np.dot(self.weights['Rz'], H[t-1]) + self.weights['bz']
            zt = np.tanh(zt_)
            Z[t] = np.copy(zt)
            
            it_ = np.dot(self.weights['Wi'], xt) + np.dot(self.weights['Ri'], H[t-1]) + self.weights['bi']
            it = sigmoid(it_)
            I[t] = np.copy(it)
            
            ot_ = np.dot(self.weights['Wo'], xt) + np.dot(self.weights['Ro'], H[t-1]) + self.weights['bo']
            ot = sigmoid(ot_)
            O[t] = np.copy(ot)
            
            ft_ = np.dot(self.weights['Wf'], xt) + np.dot(self.weights['Rf'], H[t-1]) + self.weights['bf']
            ft = sigmoid(ft_)
            F[t] = np.copy(ft)
            
            
            ct = zt * it + C[t-1] * ft
            C[t] = np.copy(ct)
            
            ht = np.tanh(ct) * ot
            
            H[t] = np.copy(ht)
            
            gt = np.dot(self.weights['Wout'], H[t]) + self.weights['bout']
            G[t] = np.copy(gt)
             
            softmaxPredictions[t] = np.copy(self.stable_softmax(gt))
        
        return softmaxPredictions, G, H, O, C, F, I, Z
    
    
    def generate(self, xt, ht_1, ct_1):
        zt_ = np.dot(self.weights['Wz'], xt) + np.dot(self.weights['Rz'], ht_1) + self.weights['bz']
        zt = np.tanh(zt_)
        
        it_ = np.dot(self.weights['Wi'], xt) + np.dot(self.weights['Ri'], ht_1) + self.weights['bi']
        it = sigmoid(it_)
        
        ot_ = np.dot(self.weights['Wo'], xt) + np.dot(self.weights['Ro'], ht_1) + self.weights['bo']
        ot = sigmoid(ot_)
        
        ft_ = np.dot(self.weights['Wf'], xt) + np.dot(self.weights['Rf'], ht_1) + self.weights['bf']
        ft = sigmoid(ft_)        
        
        ct = zt * it + ct_1 * ft
        
        ht = np.tanh(ct) * ot
        
        return ht, ct
    
    
    def calculate_loss_batch(self, inputBatch, trueOutputBatch):
        loss = 0.0
        for i in range(len(inputBatch)):
            loss += self.calculate_loss(inputBatch[i], trueOutputBatch[i])
            
        return loss / len(inputBatch)
           
                    
    def calculate_loss(self, inputSeq, trueOutputSeq):
        L = 0
        softmaxPredictions, G, H, O, C, F, I, Z = self.forwardPass(inputSeq)
        for t in range(len(softmaxPredictions)):
            L -= np.log(softmaxPredictions[t][trueOutputSeq[t]][0])
        
        return L


    def backProp(self, inputBatch, trueOutputBatch):
        deltas = {}
        dWz = np.zeros(self.weights['Wz'].shape)
        dWi = np.zeros(self.weights['Wi'].shape)
        dWf = np.zeros(self.weights['Wf'].shape)
        dWo = np.zeros(self.weights['Wo'].shape)
        dWout = np.zeros(self.weights['Wout'].shape)
        
        dRz = np.zeros(self.weights['Rz'].shape)
        dRi = np.zeros(self.weights['Ri'].shape)
        dRf = np.zeros(self.weights['Rf'].shape)
        dRo = np.zeros(self.weights['Ro'].shape)
        
        dbz = np.zeros(self.weights['bz'].shape)
        dbi = np.zeros(self.weights['bi'].shape)
        dbf = np.zeros(self.weights['bf'].shape)
        dbo = np.zeros(self.weights['bo'].shape)
        dbout = np.zeros(self.weights['bout'].shape)
        
        for b in range(len(inputBatch)): #for each input sentence in the batch
            currentInput = inputBatch[b]
            currentTrueOutput = trueOutputBatch[b]
            chunckCount = int(math.ceil(float(len(currentInput)) / self.BPTT_length))
            Ht_1 = None; Ct_1 = None
            start = 0; end = 0
            for chnk in range(chunckCount):
                start = chnk * self.BPTT_length
                end = start + self.BPTT_length 
                X = currentInput[start : end]
                Y = currentTrueOutput[start : end]
                softmaxPredictions, G, H, O, C, F, I, Z = self.forwardPass(X, Ht_1, Ct_1)
                
                Ht_1 = H[len(X)-1]; Ct_1 = C[len(X)-1]
                
                T = len(X)
                
                dG = {}; dH = {}; dO = {}; dC = {}; dF = {}; dI = {}; dZ = {}
                
                for t in reversed(range(T)): #from idx T-1 down to 0
                    if t not in dG: dG[t] = np.zeros((self.inputDim,  1))
                    if t not in dH: dH[t] = np.zeros((self.hiddenDim, 1))
                    if t not in dO: dO[t] = np.zeros((self.hiddenDim, 1))
                    if t not in dC: dC[t] = np.zeros((self.hiddenDim, 1))
                    if t not in dF: dF[t] = np.zeros((self.hiddenDim, 1))
                    if t not in dI: dI[t] = np.zeros((self.hiddenDim, 1))
                    if t not in dZ: dZ[t] = np.zeros((self.hiddenDim, 1))
                    
                    xt = np.zeros((self.inputDim,1))
                    xt[X[t]][0] = 1.0
                        
                    yt = np.zeros((self.inputDim,1))
                    yt[Y[t]][0] = 1.0
                    
                    dG[t] += softmaxPredictions[t] - yt
        
                    dH[t] += np.dot(self.weights['Wout'].T, dG[t])
                    if(t+1 < T):
                        dH[t] += np.dot(self.weights['Rz'], dZ[t+1]) + np.dot(self.weights['Ri'], dI[t+1]) + np.dot(self.weights['Rf'], dF[t+1]) + np.dot(self.weights['Ro'], dO[t+1])
                    
                    dO[t] += dH[t] * np.tanh(C[t]) * O[t] * (1 - O[t]) 
                    
                    dC[t] += dH[t] * O[t] * (1 - np.tanh(C[t]) * np.tanh(C[t]))
                    if(t+1 < T):
                        dC[t] += dC[t+1] * F[t+1]
                    
                    if(t-1 >= 0):  
                        dF[t] += dC[t] * C[t-1] * F[t] * (1 - F[t])
                    
                    dI[t] += dC[t] * Z[t] * I[t] * (1 - I[t])
                    
                    dZ[t] += dC[t] * I[t] * (1 - Z[t]*Z[t])
                    
                    dWz += np.outer(dZ[t], xt)
                    dWi += np.outer(dI[t], xt)
                    dWf += np.outer(dF[t], xt)
                    dWo += np.outer(dO[t], xt)
                    dWout += np.outer(dG[t], H[t])
                    
                    '''
                    if(t+1 < T):
                        dRz += np.outer(dZ[t+1], H[t])
                        dRi += np.outer(dI[t+1], H[t])
                        dRf += np.outer(dF[t+1], H[t])
                        dRo += np.outer(dO[t+1], H[t])
                    '''
                    
                    dRz += np.outer(dZ[t], H[t-1])
                    dRi += np.outer(dI[t], H[t-1])
                    dRf += np.outer(dF[t], H[t-1])
                    dRo += np.outer(dO[t], H[t-1])
                    
                    dbz += dZ[t]
                    dbi += dI[t]
                    dbf += dF[t]
                    dbo += dO[t]
                    dbout += dG[t]
        
        deltas['Wz'] = dWz; deltas['Wi'] = dWi; deltas['Wf'] = dWf; deltas['Wo'] = dWo
        deltas['Rz'] = dRz; deltas['Ri'] = dRi; deltas['Rf'] = dRf; deltas['Ro'] = dRo
        deltas['bz'] = dbz; deltas['bi'] = dbi; deltas['bf'] = dbf; deltas['bo'] = dbo
        deltas['Wout'] = dWout; deltas['bout'] = dbout

        return deltas
      
      
    def SGD(self, deltas):
        for d in deltas:
            #batch normalization
            deltas[d] /= float(self.batchSize)
            
            #gradient clipping
            if self.clipGradients:
                np.clip(deltas[d], -5, 5, out=deltas[d]) # clip to overcome exploding gradients
                
            if self.useAdaGrad:
                self.gradientMemory[d] += deltas[d] * deltas[d] # updating memory
                self.weights[d] -= self.learningRate * deltas[d] / np.sqrt(self.gradientMemory[d] + 1e-8)
            else:
                self.weights[d] -= self.learningRate * deltas[d]
      
                
    def stable_softmax(self, X):
        #table softmax
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)
    
        #exps = np.exp(X)
        #return exps / np.sum(exps)


    def train(self, trainingSet = [], trainingTruth = []):
        batchCount = int(math.ceil(float(len(trainingSet)) / self.batchSize))
        for b in range(batchCount):
            X_batch = trainingSet[b*self.batchSize : (b+1)*self.batchSize]
            Y_batch = trainingTruth[b*self.batchSize : (b+1)*self.batchSize]
            deltas = self.backProp(X_batch, Y_batch)
            self.SGD(deltas)
            
        self.epoch += 1
            
    


def main():
    np.random.seed(7)
    
    trainPath = 'toyExample' 
    modelPath = 'toyExample_zLSTM.pkl'
    H = 50 # LSTM inner dimension size
    epochs = 1000
    learningRate = 0.1 
    clipGradients = True
    useAdaGrad = True
    batchSize = 1
    BPTT_length = 25
    dataGenerationType = DATA_GENERATION.SAMPLING #either sample the prob dist of the output or use maxlikelihood of the output
    startSeed = '[' #starting seed char for generation
    genLength = 50 #number of char to generate
    instanceCount = 5 #number of sequences to generate
    
    X_train, Y_train, v2i, i2v = preProcessing_charBased(filePath=trainPath, v2i = None, i2v = None)
    
    
    D = len(v2i) # Number of input dimension = number of items in vocabulary
    
    print 'Input Size=%d, Hidden Size=%d' % (D, H)
    
    lstm = zLSTM(inputDim = D, 
                 hiddenDim = H, 
                 learningRate = learningRate, 
                 clipGradients = clipGradients, 
                 useAdaGrad = useAdaGrad, 
                 batchSize = batchSize, 
                 BPTT_length = BPTT_length)
    
    crossEntropyLoss = lstm.calculate_loss_batch(X_train, Y_train)
    print 'Initial Cross Entropy Loss = ', crossEntropyLoss
    
    
    print 'Starting training'
    
    for i in range(epochs):

        lstm.train(X_train, Y_train)
        
        if i % 10 == 0:
            print 'Epoch %d ##########################################################################################' % (i)
            crossEntropyLoss = lstm.calculate_loss_batch(X_train, Y_train)
            print 'Cross Entropy TRAIN Loss = ', crossEntropyLoss
            
            generatedSamples = generateText(lstm = lstm, v2i = v2i, i2v = i2v, 
                                            startSeed = startSeed, 
                                            genLength = genLength, 
                                            instanceCount = instanceCount, 
                                            dataGenerationType = dataGenerationType)
            print 'Generated data:\n', '\n'.join(generatedSamples)
            
            #pkl.dump([lstm,v2i,i2v], open(modelPath+str(i)+'.pkl', 'wb'))
            #print 'Model is saved to: ' + modelPath+str(i)+'.pkl'
            
        #shuffle the training set for every epoch
        combined = list(zip(X_train, Y_train))
        random.shuffle(combined)
        X_train[:], Y_train[:] = zip(*combined)
        
    pkl.dump([lstm,v2i,i2v], open(modelPath, 'wb'))
    print 'Model is saved to ', modelPath
    
    


if __name__ == '__main__':
    main()
    #simulateData()
    
