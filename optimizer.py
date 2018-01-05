#! /usr/bin/env python
'''
Created on March 8, 2012

@author: mllamosa
'''
__all__ = ['Ga','ga_optimizer']

import logging
import numpy as np
from math import exp, sin, cos
import math
from time import time
import scipy as sc
from scipy.stats import linregress
#from scikits.learn.metrics import roc_curve as roc, auc
from sklearn.metrics import roc_curve as roc, auc
from ga_opt import GA
from utilities import all_stats
from data_driven import Model
from scipy.stats.stats import pearsonr


class Ga():

    '''
        Class to optimize inputs and hyper-parameters of machine-leanring objects using a generic GA
    '''
    def __init__(self,ml_model):
            self.model = ml_model
            self.chromosize = self.get_chromosize()

    def fit(self,chromo):

        ''' Generate the var idx list from the chromosome and add the response variable idx to the var list'''
        nvar = self.model.nvariables
        chromo = np.array(chromo)
        subset = np.where(chromo[0:nvar]==1)[0]
        subset = np.hstack((subset,nvar))
        if len(subset)>2:

            ''' Update the data to the reduced form'''
            ''' Generating parameter values from chromosome'''
            parameters = self.model.function.get_parameters_options()
            parametervalues = dict()
            parpointer1 = nvar
            for parameterkey in  parameters.keys():
                '''Parameters are selected by idx from converting the binary gene to int
                where a binary string of lengh log(N,2) gives N parameter choices in the range [0,..,N-1]
                '''
                parpointer2 = parpointer1 + math.log(len(parameters[parameterkey]),2)
                parameterIDbinary = chromo[int(parpointer1):int(parpointer2)]
                parpointer1 = parpointer2
                parameterID = int("".join(map(str,parameterIDbinary)),2)
                parametervalues[parameterkey] = parameters[parameterkey][parameterID]

            '''Update the function with new parameters'''
            function = self.model.function.copy()
            function.set_parameters(parametervalues)
            copymodel = Model(data = self.model.data[:,subset], groups = self.model.groups, varnames = self.model.varnames[subset], partition = self.model.partition,function=function,nfo=self.model.nfo)
            copymodel.crossvalidating()
            self.newmodel = copymodel
            if self.newmodel.groups:
                            trainidx = np.array([item for idx in self.newmodel.partition.train for item in self.newmodel.groups[idx]])
            else:
                            trainidx = self.newmodel.partition.train
            return (1 - pearsonr(self.newmodel.data[trainidx,-1],self.newmodel.validation)[0])
        else:
            print ("Bad optimization, keeping same model")
            self.newmodel = self.model
            '''Return the fitness value 1-R2cv'''
            return 1

    def load_model(self,ml_model):
        self.model = ml_model

    def get_chromosize(self):
        ''' Calculating size of chromosome'''
        nvar=self.model.nvariables
        self.chromosize = nvar

        ''' Generating parameter values from chromosome'''
        try:
            parameters = self.model.function.get_parameters_options()
            for parameterkey in  parameters.keys():
                self.chromosize += math.log(len(parameters[parameterkey]),2)
        except:
            pass

        self.chromosize = int(self.chromosize)
        return self.chromosize

class ga_optimizer():
    def __init__(self,npop=100, ngen=100, bias=0.90, inipop = False):
        self.npop = npop
        self.ngen = ngen
        self.bias = bias
        self.inipop = inipop

    def run(self,model,folder=None):
        self.opt = Ga(model)
        pop=None

        if self.inipop:
            try:
                pop=np.loadtxt('iniPopulation.csv',dtype="string")
            except:
                print ('Fail  to upload initial population, generating new')
        #print self.npop,self.ngen,self.bias
        g = GA(fitfunc = self.opt, nbit=self.opt.chromosize, npop=self.npop,ngen=self.ngen,bias=self.bias,inipop=pop,runfolder=folder)
            #logging.basicConfig(filename='test.log', level=logging.DEBUG)
            #print g.best.chromo, g.best.fitness
        self.opt.fit(g.best.chromo)
        return self.opt.newmodel

class Grid():
	def __init__(self,model,parameters_options=None):
		self.model=model

