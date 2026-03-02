import numpy as np

class AdaptiveMultiPopDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size scales with dimensions
        self.mutation_factor = 0.5  # Differential weight
        self.crossover_prob = 0.9  # Crossover probability
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        # Initialize multiple subpopulations
        num_subpops = 5
        subpop_sizes = [self.pop_size // num_subpops] * num_subpops
        subpops = [self._initialize_population(bounds, size) for size in subpop_sizes]
        
        best_solution = None
        best_fitness = float('inf')

        while self.evaluations < self.budget:
            for subpop in subpops:
                # Evaluate subpopulation
                fitness = np.apply_along_axis(func, 1, subpop)
                self.evaluations += len(fitness)

                # Update global best
                local_best_idx = np.argmin(fitness)
                local_best = subpop[local_best_idx]
                local_best_fitness = fitness[local_best_idx]

                if local_best_fitness < best_fitness:
                    best_fitness = local_best_fitness
                    best_solution = local_best

                # Perform differential evolution
                subpop = self._differential_evolution(subpop, fitness, bounds, func)

        return best_solution

    def _initialize_population(self, bounds, size):
        return bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(size, self.dim)

    def _differential_evolution(self, pop, fitness, bounds, func):
        new_pop = np.copy(pop)
        for i in range(len(pop)):
            idxs = [idx for idx in range(len(pop)) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            centroid = np.mean(pop, axis=0)  # Calculate centroid
            mutant = a + self.mutation_factor * (b - c + centroid - pop[i])  # Use centroid in mutation
            mutant = np.clip(mutant, bounds[0], bounds[1])
            
            cross_points = np.random.rand(self.dim) < self.crossover_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            
            trial = np.where(cross_points, mutant, pop[i])
            trial_fitness = func(trial)
            self.evaluations += 1
            
            if trial_fitness < fitness[i]:
                new_pop[i] = trial
                fitness[i] = trial_fitness
        
        return new_pop