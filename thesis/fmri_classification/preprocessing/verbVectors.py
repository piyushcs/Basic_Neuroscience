import requests
import numpy
from scipy.io import loadmat
from bs4 import BeautifulSoup
import numpy as np

# fmri cmu verb vectors class
class semanticVectors:
    
    def __init__(self, pageUri=None):
        self.__pageUri = pageUri

        if pageUri == None:
            self.__pageUri = 'http://www.cs.cmu.edu/~tom/science2008/semanticFeatureVectors.html'
        
        self.__bs4 = None
        self.nounLabels = None
        self.verbVectors = None
        
        # initialise
        
        self.__scrapPage()
        self.__parseNouns()
        self.__parseVerbVectors()
    
    def __scrapPage(self):
        # downloading the data from web page
        pageData = requests.get(self.__pageUri)
        soup = BeautifulSoup(pageData.text, 'html.parser')
        
        self.__bs4 = soup

    def __parseNouns(self):
        # find Noun labels
        nounLabelsData = self.__bs4.find_all('a')

        nounLabels = []
        for nld in nounLabelsData:
            attr = nld.attrs
            if 'name' in attr and nld['name'] != 'top':
                nounLabels.append(nld['name'])
        
        self.nounLabels = nounLabels
    
    def __parseVerbVectors(self):
        # find verb labels
        verbLabelsData = self.__bs4.find_all('ul')
        
        # initializing verbs
        verbs = dict()
        for vld in verbLabelsData[:1]:
            vrbData = vld.text.split('\n')[1:-1]
            for vr in vrbData:
                verbs[vr.split()[0]] = float(0.0)
        
        # creating verb vectors
        verbVectors = list()
        for v in verbLabelsData:
            vrbData = v.text.split('\n')[1:-1]
            for vr in vrbData:
                verbLabel = vr.split()[0]
                verbValue = vr.split()[-1].split(')')[0][1:]
                verbs[verbLabel] = float(verbValue)

            verbVectors.append(list(verbs.values()))
            for vr in vrbData:
                verbs[vr.split()[0]] = float(0.0)
        
        self.verbVectors = np.array(verbVectors)
