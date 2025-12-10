import numpy as np

class EnhancedMultiAgentHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        population_size = 10 + 2 * self.dim
        F_base = 0.5  # Base differential weight
        CR_base = 0.9  # Base crossover probability
        local_search_prob = 0.2  # Probability of performing local search
        
        # Initialize population
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.rand(population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        # Initialize adaptive learning rate factor
        learning_rate = 0.1

        while evaluations < self.budget:
            # Calculate population diversity
            population_std = np.std(population, axis=0)
            diversity = np.mean(population_std / (ub - lb))

            # Adapt mutation and crossover rates based on diversity
            F = F_base + learning_rate * np.tanh(5 * (0.5 - diversity))
            CR = CR_base - learning_rate * np.tanh(5 * (0.5 - diversity))

            # Adaptive swarm factor to enhance exploration
            swarm_factor = 0.5 + 0.5 * np.sin(2 * np.pi * evaluations / self.budget)
            
            for i in range(population_size):
                # Ensemble mutation strategies
                idx_pool = list(range(population_size))
                idx_pool.remove(i)
                a, b, c = np.random.choice(idx_pool, 3, replace=False)

                # Combining different mutation strategies
                mutant1 = population[a] + F * (population[b] - population[c])
                mutant2 = population[a] - F * (population[b] - population[c])
                mutant = np.where(np.random.rand(self.dim) < 0.5, mutant1, mutant2)
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

                # Enhanced Adaptive local search
                if np.random.rand() < local_search_prob and evaluations < self.budget:
                    adaptive_step_size = (ub - lb) * 0.1 * (0.5 + 0.5 * (fitness.mean() - fitness[i]) / (fitness.std() + 1e-12))
                    local_trial = population[i] + np.random.uniform(-adaptive_step_size, adaptive_step_size)
                    local_trial = np.clip(local_trial, lb, ub)
                    local_fitness = func(local_trial)
                    evaluations += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_trial
                        fitness[i] = local_fitness

                if evaluations >= self.budget:
                    break

            # Update learning rate factor based on fitness improvement
            fitness_improvement = np.abs(fitness[i] - trial_fitness) / (fitness[i] + 1e-12)
            learning_rate = 0.1 + 0.05 * np.clip(fitness_improvement, 0, 1)

        # Return the best found solution
        best_index = np.argmin(fitness)
        return population[best_index]