import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Initialize population
        population_size = 10
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # Adaptive Differential Evolution parameters
        F_min, F_max = 0.5, 1.0  # Range for differential weight
        CR_min, CR_max = 0.4, 0.9  # Range for crossover probability

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Adaptive parameters
                F = F_min + (F_max - F_min) * evaluations / self.budget
                CR = CR_max - (CR_max - CR_min) * evaluations / self.budget

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

            # Local Search phase
            if evaluations < self.budget and np.random.rand() < 0.2:  # 20% chance of local search
                best_idx = np.argmin(fitness)
                local_candidate = population[best_idx] + np.random.normal(0, 0.1, self.dim)
                local_candidate = np.clip(local_candidate, func.bounds.lb, func.bounds.ub)
                local_fitness = func(local_candidate)
                evaluations += 1

                if local_fitness < fitness[best_idx]:
                    population[best_idx] = local_candidate
                    fitness[best_idx] = local_fitness

            # Random Search
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