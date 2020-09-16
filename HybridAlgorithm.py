# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:40:39 2020

@author: Rakin Shahriar
"""

from surprise import AlgoBase

class HybridAlgorithm(AlgoBase):
    def __init__(self, algorithms, weights, sim_options=[]):
        AlgoBase.__init__(self)
        self.algorithms = algorithms
        self.weights = weights
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        for algorithm in self.algorithms:
            algorithm.fit(trainset)
            
        return self
    
    def estimate(self, u, i):
        sumScores = 0
        sumWieghts = 0
        for idx in range(len(self.algorithms)):
            sumScores += self.algorithms[idx].estimate(u,i)* self.weights[idx]
            sumWieghts += self.weights[idx]
            
        return sumScores / sumWieghts