import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        population_size = 10 + 2 * self.dim
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        local_search_prob = 0.2  # Probability of performing local search
        
        # Initialize population
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.rand(population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            for i in range(population_size):
                # Differential mutation
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Adaptive local search
                if np.random.rand() < local_search_prob and evaluations < self.budget:
                    step_size = (ub - lb) * 0.05
                    local_trial = population[i] + np.random.uniform(-step_size, step_size)
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    evaluations += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_trial
                        fitness[i] = local_fitness

                if evaluations >= self.budget:
                    break

        # Return the best found solution
        best_index = np.argmin(fitness)
        return population[best_index]