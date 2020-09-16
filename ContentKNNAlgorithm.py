# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:40:05 2020

@author: Rakin Shahriar
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
from MovieLens import MovieLens
import math
import numpy as np
import heapq

class ContentKNNAlgorithm(AlgoBase):
    
    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        ml = MovieLens()
        genres = ml.getGenres()
        years = ml.getYears()
        mes = ml.getMiseEnScene()
        
        print("Computing content-based similarity matrix...")
        
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        
        for thisRating in range(self.trainset.n_items):
            if(thisRating % 100 == 0):
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating+1, self.trainset.n_items):
                thisMovieID = int(self.trainset.to_raw_iid(thisRating))
                otherMovieID = int(self.trainset.to_raw_iid(otherRating))
                genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                yeaarSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
                mesSimilarity = self.computeMiseEnSceneSimilarity(thisMovieID, otherMovieID, mes)
                self.similarities[thisRating, otherRating] = genreSimilarity * yeaarSimilarity * mesSimilarity
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
        
        print("...done.")
        return self
    
    def computeGenreSimilarity(self, movie1, movie2, genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy = 0,0,0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)
    
    def computeYearSimilarity(self, movie1, movie2, years):
        diff = abs(years[movie1] - years[movie2])
        sim = math.exp(-diff /10.0)
        return sim
    
    def computeMiseEnSceneSimilarity(self, movie1, movie2, mes):
        mes1 = mes[movie1]
        mes2 = mes[movie2]
        if(mes1 and mes2):
            shotLengthDiff = math.fabs(mes1[0] - mes2[0])
            colorVarianceDiff = math.fabs(mes1[1] - mes2[1])
            motionDiff = math.fabs(mes1[3] - mes2[3])
            lightingDiff = math.fabs(mes1[5] - mes2[5])
            numShotsdiff = math.fabs(mes1[6] - mes2[6])
            return shotLengthDiff * colorVarianceDiff * motionDiff * lightingDiff * numShotsdiff
        else:
            return 0
    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')
            
        neighbours = []
        for rating in self.trainset.ur[u]:
            genreSimilarity = self.similarities[i,rating[0]]
            neighbours.append((genreSimilarity, rating[1]))
            
        k_neighbours = heapq.nlargest(self.k, neighbours, key = lambda t : t[0])
        
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbours:
            if(simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
            
        if(simTotal == 0):
            raise PredictionImpossible('No neighbours')
            
        predictedRating = weightedSum / simTotal
        
        return predictedRating