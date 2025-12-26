import numpy as np

class HybridOptimizer:
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
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability

        def levy_flight(Lambda):
            # Levy flight formula
            sigma1 = np.power((np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) /
                              (np.math.gamma((1 + Lambda) / 2) * Lambda * np.power(2, ((Lambda - 1) / 2))), (1 / Lambda))
            u = np.random.normal(0, sigma1, size=self.dim)
            v = np.random.normal(0, 1, size=self.dim)
            step = u / np.power(np.abs(v), (1 / Lambda))
            return step

        while evaluations < self.budget:
            best_idx = np.argmin(fitness)  # Track best solution index
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)

                # Levy flight mutation
                if np.random.rand() < 0.5:
                    mutant += levy_flight(1.5)

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

            # Elitism: Preserve best solution
            if evaluations < self.budget:
                rand_ind = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                rand_fitness = func(rand_ind)
                evaluations += 1

                if rand_fitness < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    fitness[worst_idx] = rand_fitness
                    population[worst_idx] = rand_ind

            # Ensure best solution is retained
            population[best_idx] = population[np.argmin(fitness)]
            fitness[best_idx] = min(fitness)

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]