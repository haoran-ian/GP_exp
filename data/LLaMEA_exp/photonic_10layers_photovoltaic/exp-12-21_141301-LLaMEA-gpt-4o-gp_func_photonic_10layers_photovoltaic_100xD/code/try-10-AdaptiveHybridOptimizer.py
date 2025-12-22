import numpy as np

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        pop_size = 20
        F_init = 0.8  # Initial differential weight
        CR_init = 0.9  # Initial crossover probability
        T0 = 1.0  # Initial temperature for simulated annealing
        alpha = 0.98  # Cooling rate
        elitism_rate = 0.1
        immigrant_rate = 0.05  # Rate of random immigrants
        min_temp_factor = 0.1  # Minimum temperature factor

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

        while num_evaluations < self.budget:
            # Dynamic adjustment of parameters
            F = F_init * (1 - num_evaluations / self.budget)
            CR = CR_init * (1 - num_evaluations / self.budget)
            temperature = max(T0 * alpha**(num_evaluations / pop_size), min_temp_factor * T0)

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
                mutant = x0 + F * (x1 - x2)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                num_evaluations += 1

                # Simulated Annealing with dynamic temperature
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                else:
                    acceptance_prob = np.exp((fitness[i] - trial_fitness) / temperature)
                    if np.random.rand() < acceptance_prob:
                        population[i] = trial
                        fitness[i] = trial_fitness

                if num_evaluations >= self.budget:
                    break

            # Reintroduce elites and random immigrants to maintain diversity
            population[:elite_count] = elites
            num_immigrants = int(immigrant_rate * pop_size)
            immigrants = np.random.rand(num_immigrants, self.dim) * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
            population[-num_immigrants:] = immigrants

        return best_solution