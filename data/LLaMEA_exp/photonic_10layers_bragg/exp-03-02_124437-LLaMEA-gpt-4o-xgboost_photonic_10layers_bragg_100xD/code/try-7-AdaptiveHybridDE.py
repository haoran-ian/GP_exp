import numpy as np

class AdaptiveHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size scales with dimensions
        self.initial_mutation_factor = 0.5  # Differential weight
        self.crossover_prob = 0.9  # Crossover probability
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        num_subpops = 5
        subpop_sizes = [self.pop_size // num_subpops] * num_subpops
        subpops = [self._initialize_population(bounds, size) for size in subpop_sizes]

        best_solution = None
        best_fitness = float('inf')

        while self.evaluations < self.budget:
            for subpop in subpops:
                fitness = np.apply_along_axis(func, 1, subpop)
                self.evaluations += len(fitness)

                local_best_idx = np.argmin(fitness)
                local_best = subpop[local_best_idx]
                local_best_fitness = fitness[local_best_idx]

                if local_best_fitness < best_fitness:
                    best_fitness = local_best_fitness
                    best_solution = local_best

                mutation_factor = self._adapt_mutation_factor(fitness)
                subpop = self._differential_evolution(subpop, fitness, bounds, func, best_solution, mutation_factor)

                if self.evaluations < self.budget:
                    subpop, fitness = self._stochastic_local_search(subpop, fitness, bounds, func)

        return best_solution

    def _initialize_population(self, bounds, size):
        return bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(size, self.dim)

    def _differential_evolution(self, pop, fitness, bounds, func, best_solution, mutation_factor):
        new_pop = np.copy(pop)
        for i in range(len(pop)):
            idxs = [idx for idx in range(len(pop)) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + mutation_factor * (b - c) + 0.1 * (best_solution - pop[i])
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

    def _stochastic_local_search(self, pop, fitness, bounds, func):
        for i in range(len(pop)):
            perturbation = np.random.normal(0, 0.1, self.dim) * (bounds[1] - bounds[0])
            perturbed = np.clip(pop[i] + perturbation, bounds[0], bounds[1])
            perturbed_fitness = func(perturbed)
            self.evaluations += 1
            
            if perturbed_fitness < fitness[i]:
                pop[i] = perturbed
                fitness[i] = perturbed_fitness
        
        return pop, fitness

    def _adapt_mutation_factor(self, fitness):
        fitness_std = np.std(fitness)
        if fitness_std < 1e-5:
            return self.initial_mutation_factor * 0.5
        elif fitness_std < 1e-3:
            return self.initial_mutation_factor
        else:
            return self.initial_mutation_factor * 1.5