import numpy as np

class RefinedHybridOptimizer:
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
        F_base = 0.6  # Base differential weight
        CR_base = 0.7  # Base crossover probability

        while evaluations < self.budget:
            # Adaptation of F and CR based on diversity
            diversity = np.mean(np.std(population, axis=0))
            F = F_base * (1 + (diversity / self.dim))
            CR = CR_base * (1 - (diversity / self.dim))

            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Enhanced Random Search
            if evaluations < self.budget:
                elite_idx = np.argmin(fitness)
                elite = population[elite_idx]
                rand_ind = np.random.normal(elite, 0.1 * (func.bounds.ub - func.bounds.lb))
                rand_ind = np.clip(rand_ind, func.bounds.lb, func.bounds.ub)
                rand_fitness = func(rand_ind)
                evaluations += 1

                if rand_fitness < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    fitness[worst_idx] = rand_fitness
                    population[worst_idx] = rand_ind

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]