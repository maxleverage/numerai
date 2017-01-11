#!/usr/bin/env python

import math as m 
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

def power_fitness(cv_scores, exponent):
	# Takes as input cross_validation log-loss and generates a discrete distribution
	return [(cv ** exponent) / np.sum(np.power(cv_scores, exponent)) for cv in cv_scores]

def rank_fitness(cv_scores, exponent):
	n = len(cv_scores)
	rank_scores = np.zeros(n)
	for i in range(n):
		rank_scores[np.argmax(cv_scores)] = n - i
		cv_scores[np.argmax(cv_scores)] = np.min(cv_scores) - 1.
	return [np.power(rank, exponent) / np.sum(np.power(rank_scores, exponent)) for rank in rank_scores]

def GA(tr_data, target, population_size, phenotype_length, n_iter, mutation_rate, fitness_function, model):
	# Inputs are population size, length of phenotype
	# Initialise initial population
	n_row, n_col = tr_data.shape
	current_gen = np.random.binomial(1, 0.5, population_size*phenotype_length).reshape((population_size, phenotype_length))
	best_score, best_solution = [], []
	for i in range(n_iter):
		print("Running %0.0f iteration" % (i+1))
		cv_score = []
		for i in range(population_size):
			diag_matrix = np.diag(current_gen[i,:])
			new_data = np.dot(tr_data, diag_matrix)
			cv_score.append(cross_val_score(model, new_data, target, cv=5, scoring='neg_log_loss').mean())
			print(cv_score[-1])
		# Saving best scores
		best_score.append(cv_score[np.argmax(cv_score)])
		best_solution.append(current_gen[np.argmax(cv_score), :])
		# Calculate fitness for breeding
		breed_dist = fitness_function(cv_score, exponent=3)
		# Draw n/2 mothers and n/2 fathers
		mothers = np.random.choice(range(population_size), size=int(population_size/2), p=breed_dist)
		fathers = np.random.choice(range(population_size), size=int(population_size/2))
		next_gen = np.zeros((population_size, n_col))
		counter = 0
		for i, j in zip(current_gen[mothers], current_gen[fathers]):
			# Perform crossover
			cross = np.random.randint(0, n_col)
			offspring1 = np.r_[i[:cross], j[cross:]]
			offspring2 = np.r_[j[:cross], i[cross:]]
			# Perform mutation
			mutation1 = np.random.binomial(1, mutation_rate, size=n_col).astype(bool)
			offspring1[mutation1] = np.abs(offspring1[mutation1] - 1)
			mutation2 = np.random.binomial(1, mutation_rate, size=n_col).astype(bool)
			offspring2[mutation2] = np.abs(offspring2[mutation2] - 1)
			next_gen[counter, :] = offspring1
			next_gen[counter+1, :] = offspring2
			counter += 2
		current_gen = next_gen
	return best_score, best_solution

