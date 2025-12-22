import numpy as np

class ImprovedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        initial_pop_size = 20
        F_init = 0.8  # Initial differential weight
        CR_init = 0.9  # Initial crossover probability
        T0 = 1.0  # Initial temperature for simulated annealing
        alpha = 0.98  # Cooling rate
        elitism_rate = 0.15  # Elitism rate
        memory_size = 10  # Memory size for adaptive learning
        memory = np.full(memory_size, F_init)  # Memory for differential weight
        
        # New dynamic crossover strategy factor
        beta = 0.5

        # Initialize population
        population = np.random.rand(initial_pop_size, self.dim)
        for i in range(initial_pop_size):
            population[i] = func.bounds.lb + population[i] * (func.bounds.ub - func.bounds.lb)
        
        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = initial_pop_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        temperature = T0
        memory_idx = 0

        # Adaptive population reduction
        reduction_factor = 0.95

        while num_evaluations < self.budget:
            pop_size = max(4, int(initial_pop_size * reduction_factor))

            # Adaptive learning rate using memory
            if np.random.rand() < 0.3:
                F = memory[memory_idx]
            else:
                # Changed line
                F = F_init * (1 - (num_evaluations / (self.budget * 2.0))) + 0.15

            # Dynamic crossover strategy
            CR = CR_init * (1 - num_evaluations / self.budget) * beta
            
            memory[memory_idx] = F
            memory_idx = (memory_idx + 1) % memory_size

            # Elitism: protect the top solutions
            elite_count = int(elitism_rate * pop_size)
            elite_indices = fitness.argsort()[:elite_count]
            elites = population[elite_indices]

            for i in range(pop_size):
                if i in elite_indices:
                    continue

                # Differential Evolution mutation and crossover
                idxs = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[idxs]
                mutant = x0 + F * (x1 - x2)  # Changed line
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Add noise reduction
                trial = 0.95 * trial + 0.05 * population[i]

                # Evaluate trial vector
                trial_fitness = func(trial)
                num_evaluations += 1

                # Simulated Annealing acceptance
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                else:
                    # Changed line
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / (temperature + 0.1))
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

                # Changed line
                if num_evaluations >= self.budget - 1:
                    break

            # Reintroduce elites to maintain diversity
            population[:elite_count] = elites

            temperature *= alpha
            # Changed line
            reduction_factor *= 0.985  # Gradually reduce population size

        return best_solution