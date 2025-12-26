import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Initialize population with dynamic scaling based on budget and dimension
        initial_pop_scale = 0.1  # Initial population scaling factor
        population_size = max(10, int(self.dim * initial_pop_scale))
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # Differential Evolution parameters with adaptive mutation factor
        F_min, F_max = 0.5, 1.0  # Differential weight range
        CR = 0.9  # Crossover probability

        while evaluations < self.budget:
            # Adaptive population size scaling
            curr_pop_size = population_size + (self.budget - evaluations) // (self.dim * 2)
            curr_pop_size = min(curr_pop_size, self.budget - evaluations)

            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Adaptive mutation factor
                F = F_min + (F_max - F_min) * (1 - evaluations / self.budget)

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

            # Strategic Random Search
            if evaluations < self.budget:
                rand_inds = np.random.uniform(func.bounds.lb, func.bounds.ub, (curr_pop_size, self.dim))
                rand_fitnesses = np.array([func(ind) for ind in rand_inds])
                evaluations += curr_pop_size

                for j, rand_fitness in enumerate(rand_fitnesses):
                    if rand_fitness < fitness[np.argmax(fitness)]:
                        worst_idx = np.argmax(fitness)
                        fitness[worst_idx] = rand_fitness
                        population[worst_idx] = rand_inds[j]

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]