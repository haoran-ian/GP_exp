import numpy as np

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize population
        population_size = 10
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # Differential Evolution parameters
        F_base = 0.5  # Starting differential weight
        CR_base = 0.9  # Starting crossover probability

        # Simulated Annealing parameters
        T_initial = 100.0  # Initial temperature
        cooling_rate = 0.99

        while evaluations < self.budget:
            T = T_initial * cooling_rate ** (evaluations // population_size)
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Adaptively update F and CR
                F = F_base + np.random.rand() * 0.3
                CR = CR_base + np.random.rand() * 0.1

                # Mutation
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])

                # Selection with Simulated Annealing
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / T):
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Random Search with adaptive probability
            if evaluations < self.budget:
                rand_ind = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                rand_fitness = func(rand_ind)
                evaluations += 1

                if rand_fitness < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    fitness[worst_idx] = rand_fitness
                    population[worst_idx] = rand_ind

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]