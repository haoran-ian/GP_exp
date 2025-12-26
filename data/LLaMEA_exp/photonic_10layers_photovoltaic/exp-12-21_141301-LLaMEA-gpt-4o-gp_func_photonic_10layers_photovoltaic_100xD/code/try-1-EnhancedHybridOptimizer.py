import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        pop_size = 20
        F = 0.8  # Initial differential weight
        CR = 0.9  # Initial crossover probability
        T0 = 1.0  # Initial temperature for simulated annealing
        alpha = 0.99  # Cooling rate
        restart_threshold = 0.2 * self.budget

        # Initialize population
        population = np.random.rand(pop_size, self.dim)
        for i in range(pop_size):
            population[i] = func.bounds.lb + population[i] * (func.bounds.ub - func.bounds.lb)
        
        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = pop_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        temperature = T0
        evaluations_since_improvement = 0

        while num_evaluations < self.budget:
            for i in range(pop_size):
                # Adapt F and CR over time
                F = 0.5 + 0.5 * np.random.rand()
                CR = 0.5 + 0.5 * np.random.rand()
                
                # Differential Evolution mutation and crossover
                idxs = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                num_evaluations += 1

                # Simulated Annealing acceptance
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    evaluations_since_improvement = 0
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / temperature)
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        evaluations_since_improvement = 0
                    else:
                        evaluations_since_improvement += 1

                if num_evaluations >= self.budget:
                    break

            # Restart mechanism
            if evaluations_since_improvement >= restart_threshold:
                # Reinitialize population except for the best solution
                for j in range(pop_size):
                    if j != best_idx:
                        population[j] = func.bounds.lb + np.random.rand(self.dim) * (func.bounds.ub - func.bounds.lb)
                fitness = np.array([func(ind) for ind in population])
                num_evaluations += pop_size
                evaluations_since_improvement = 0

            temperature *= alpha

        return best_solution