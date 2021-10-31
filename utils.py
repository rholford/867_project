#Preprocessing mockup (utils?)
import csv
import json
import ast
import random

#Need keyword/vector?
#GPU select?

#read data method?
#def...


def ExtractData(datasetName):
    """
    Returns training and testing dataset from dataset name ('gab, 'reddit', 'conan')
    """
    #training data percent
    pct = 0.80
    if datasetName in ['reddit', 'gab']:
        dataFile = open ('./data/ohs/' + datasetName + '.csv', 'r', encoding='utf-8')
        hateSpeechBlob = []
        hsIdx = []
        responseBlob = []
        
        reader = csv.DictReader(dataFile)
        for row in reader:
            x = row['text']
            y = row['hate_speech_idx']
            z = row['response']
            if y == 'n/a':
                continue
            hateSpeechBlob.append(x)
            hsIdx.append(y)
            responseBlob.append(z)
        #print(hateSpeechBlob[1])   
        hateCount = 0
        for item in hsIdx:
            for i in item.strip('[]').split(', '):
                hateCount += 1

        hateSpeech, counterSpeech = [], []
        lineNumber = 0
        for hs, idx, cs in zip(hateSpeechBlob, hsIdx, responseBlob):
            hs = hs.strip().split('\n')
            for i in idx.strip('[]').split(', '):
                try:
                    hateSpeech.append('. '.join(hs[int(i) - 1].split('. ')[1:]).strip('\t'))
                except:
                    continue
                    #Note this is because there is an error in the data that throws out of bounds
                temp = []
                for j in splitResponse(cs):
                    if j.lower() == 'n/a':
                        continue
                    temp.append(j)
                counterSpeech.append(temp)
                lineNumber += 1
        hateCount = len(hateSpeech)
                

    elif datasetName == 'conan':
        dataFile = open ('./data/conan/CONAN.json', 'r')
        fileText = []
        for line in dataFile:
            #print(line)
            fileText.append(json.loads(line))

        enText =  []
        for item in fileText[0]['conan']:
            if (item['cn_id'][:2] == 'EN'):
                enText.append(item)

        hateSpeech = []
        counterSpeech =[] 
        for item in enText:
            hateSpeech.append(item['hateSpeech'].strip())
            counterSpeech.append([item['counterSpeech'].strip()])
        hateCount = len(hateSpeech)
        dataFile.close()
       
    #print(hateSpeech)
    #split the data randomly into train/test
    randomIndex = []
    for num in range(hateCount):
        randomIndex.append(num) 
    random.shuffle(randomIndex)
    trainIndex = sorted(randomIndex[:int(pct*len(randomIndex))])
    trainHate = []
    #print(len(hateSpeech))
    #print(hateCount)
    #print(len(counterSpeech))
    for i in range(hateCount):
        if (i in trainIndex):
            trainHate.append(hateSpeech[i])
    trainCounter = []
    for i in range(hateCount):
        if (i in trainIndex):
            trainCounter.append(counterSpeech[i])

    testHate = []
    for i in range(hateCount):
        if (i not in trainIndex):
            testHate.append(hateSpeech[i])
    testCounter = []
    for i in range(hateCount):
        if (i not in trainIndex):
            testCounter.append(counterSpeech[i])
    return trainHate, trainCounter, testHate, testCounter

#helper function for csvs
def splitResponse(strResp):
    result = ast.literal_eval(strResp)
    #print(result)
    retVal = []
    for item in result:
        retVal.append(item)
    return retVal

#context to response goes here
#def...

#model code goes here, see what we need
#def///

def main():
    #a, b, c, d = ExtractData('conan')
    #a, b, c, d= ExtractData('reddit')
    a,b,c,d = ExtractData('gab')
    #print(a[3])
    #print(b[3])
    print(len(a) + len(c))
    print(len(b) + len(d))
if __name__ == "__main__":
    main()
