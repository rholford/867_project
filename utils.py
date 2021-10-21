#Preprocessing mockup (utils?)
import csv
import json


def ExtractData(datasetName):
    if datasetName == ('reddit' or 'gab'):
        conanFile = open ('./data/conan/CONAN.json', 'r')
    
    if datasetName == 'conan':
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

        #todo: shuffle, split into test/train
        return hateSpeech, counterSpeech

def main():
    ExtractData('conan')

if __name__ == "__main__":
    main()
