# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:56:35 2020

@author: Rakin Shahriar
"""

from RecommenderMatrics import RecommenderMatrics
from EvaluationData import EvaluationData

class EvaluatedAlgorithm:
    def __init__(self,algorithm,name):
        self.algorithm = algorithm
        self.name = name
        
    def Evaluate(self,evaluationData, doTopN, n=10, verbose =True):
        metrics = {}
        if (verbose):
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluationData.GetTrainSet())
        predictions = self.algorithm.test(evaluationData.GetTestSet())
        metrics["RMSE"] = RecommenderMatrics.RMSE(predictions)
        metrics["MAE"] = RecommenderMatrics.MAE(predictions)
        if(doTopN):
            if(verbose):
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(evaluationData.GetLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(evaluationData.GetLOOCVTestSet())
            allPredictions = self.algorithm.test(evaluationData.GetLOOCVTestSet())
            topNPredicted = RecommenderMatrics.GetTopN(allPredictions, n)
            if(verbose):
                print("Computing hit-rate and rank metrics...")
                
            metrics["HR"] = RecommenderMatrics.HitRate(topNPredicted, leftOutPredictions)
            metrics["cHR"] = RecommenderMatrics.CumulativeHitRate(topNPredicted, leftOutPredictions)
            metrics["ARHR"] = RecommenderMatrics.AverageReciprocalHitRank(topNPredicted, leftOutPredictions)
            
            if(verbose):
                print("Computing recommendation s with full dataset...")
            self.algorithm.fit(evaluationData.GetFullTrainSet())
            allPredictions = self.algorithm.test(evaluationData.GetFullAntiTestSet())
            topNPredicted = RecommenderMatrics.GetTopN(allPredictions, n)
            if(verbose):
                print("Analyzing coverage, diversity and novelty...")
                
            metrics["Coverage"] = RecommenderMatrics.UserCoverage( topNPredicted, evaluationData.GetFullTrainSet().n_users,
                                                                  ratingThreshold = 4.0)
            metrics["Diversity"] = RecommenderMatrics.Diversity(topNPredicted, evaluationData.GetSimilarities())
            metrics["Novelty"] = RecommenderMatrics.Novelty(topNPredicted,
                                                            evaluationData.GetPopularityRankings())
            
        if(verbose):
            print("Analysis complete.")
            
        return metrics
    
    def GetName(self):
        return self.name
    
    def GetAlgorithm(self):
        return self.algorithm