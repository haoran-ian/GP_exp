import numpy as np

class EnhancedAdaptiveMultiAgentHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        population_size = 10 + 2 * self.dim
        F_base = 0.5
        CR_base = 0.9
        local_search_prob = 0.2
        
        # Initialize population
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.rand(population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        # Niching parameters
        niche_radius = 0.1 * np.linalg.norm(ub - lb) / np.sqrt(self.dim)

        while evaluations < self.budget:
            # Calculate population diversity
            population_std = np.std(population, axis=0)
            diversity = np.mean(population_std / (ub - lb))

            # Adapt mutation and crossover rates based on diversity
            F = F_base + 0.5 * np.tanh(5 * (0.5 - diversity))
            CR = CR_base - 0.3 * np.tanh(5 * (0.5 - diversity))

            # Dynamic self-adaptive learning
            learning_rate = 0.1 + 0.4 * diversity

            # Adaptive swarm factor
            swarm_factor = 0.5 + 0.5 * np.sin(2 * np.pi * evaluations / self.budget)

            for i in range(population_size):
                # Differential mutation with adaptive F
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = swarm_factor * (population[a] + F * (population[b] - population[c]))
                mutant = np.clip(mutant, lb, ub)

                # Crossover with adaptive CR
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Enhanced Adaptive local search with niching
                if np.random.rand() < local_search_prob and evaluations < self.budget:
                    adaptive_step_size = (ub - lb) * learning_rate * (0.5 + 0.5 * (fitness.mean() - fitness[i]) / (fitness.std() + 1e-12))
                    local_trial = population[i] + np.random.uniform(-adaptive_step_size, adaptive_step_size)
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    evaluations += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_trial
                        fitness[i] = local_fitness

                # Niching mechanism
                for j in range(population_size):
                    if i != j and np.linalg.norm(population[i] - population[j]) < niche_radius:
                        population[j] = lb + np.random.rand(self.dim) * (ub - lb)
                        fitness[j] = func(population[j])
                        evaluations += 1

                if evaluations >= self.budget:
                    break

        # Return the best found solution
        best_index = np.argmin(fitness)
        return population[best_index]