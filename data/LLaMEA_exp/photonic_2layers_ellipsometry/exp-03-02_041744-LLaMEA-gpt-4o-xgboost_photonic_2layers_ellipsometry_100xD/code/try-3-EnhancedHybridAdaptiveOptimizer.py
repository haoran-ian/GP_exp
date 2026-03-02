import numpy as np

class EnhancedHybridAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        
    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        initial_pop_size = 10 * self.dim  # Initial population size
        de_cr = 0.9  # DE crossover probability
        simplex_size = self.dim + 1  # Size of Nelder-Mead simplex

        # Initialize population for DE
        population = np.random.uniform(bounds[0], bounds[1], (initial_pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += initial_pop_size

        def dynamic_population_size():
            nonlocal population, fitness
            if self.evaluations > self.budget * 0.5:
                target_pop_size = int(initial_pop_size * 0.5)
            else:
                target_pop_size = initial_pop_size
            if len(population) > target_pop_size:
                indices = np.argsort(fitness)[:target_pop_size]
                population = population[indices]
                fitness = fitness[indices]

        def adaptive_mutation():
            return 0.8 * (1 - (self.evaluations / self.budget))  # Decrease mutation factor over time

        def differential_evolution():
            nonlocal population, fitness
            de_f = adaptive_mutation()
            for _ in range(self.budget - self.evaluations):
                if self.evaluations >= self.budget:
                    break
                for i in range(len(population)):
                    indices = np.random.choice(len(population), 3, replace=False)
                    x0, x1, x2 = population[indices]
                    mutant = np.clip(x0 + de_f * (x1 - x2), bounds[0], bounds[1])
                    cross_points = np.random.rand(self.dim) < de_cr
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                    f_trial = func(trial)
                    self.evaluations += 1
                    if f_trial < fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial

        def nelder_mead():
            nonlocal population, fitness
            for _ in range(self.budget - self.evaluations):
                if self.evaluations >= self.budget:
                    break
                simplex_indices = np.argsort(fitness)[:simplex_size]
                simplex = population[simplex_indices]
                simplex_fitness = fitness[simplex_indices]

                centroid = np.mean(simplex[:-1], axis=0)
                reflection = np.clip(centroid + (centroid - simplex[-1]), bounds[0], bounds[1])
                reflection_fitness = func(reflection)
                self.evaluations += 1

                if reflection_fitness < simplex_fitness[0]:
                    expansion = np.clip(centroid + 2 * (reflection - centroid), bounds[0], bounds[1])
                    expansion_fitness = func(expansion)
                    self.evaluations += 1

                    if expansion_fitness < reflection_fitness:
                        simplex[-1] = expansion
                        simplex_fitness[-1] = expansion_fitness
                    else:
                        simplex[-1] = reflection
                        simplex_fitness[-1] = reflection_fitness
                elif reflection_fitness < simplex_fitness[-2]:
                    simplex[-1] = reflection
                    simplex_fitness[-1] = reflection_fitness
                else:
                    contraction = np.clip(centroid + 0.5 * (simplex[-1] - centroid), bounds[0], bounds[1])
                    contraction_fitness = func(contraction)
                    self.evaluations += 1

                    if contraction_fitness < simplex_fitness[-1]:
                        simplex[-1] = contraction
                        simplex_fitness[-1] = contraction_fitness
                    else:
                        simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                        simplex_fitness[1:] = [func(simplex[i]) for i in range(1, simplex_size)]
                        self.evaluations += simplex_size - 1

                population[simplex_indices] = simplex
                fitness[simplex_indices] = simplex_fitness

        # Hybrid strategy with dynamic adjustments
        iteration = 0
        while self.evaluations < self.budget:
            dynamic_population_size()
            if iteration % 2 == 0:
                differential_evolution()
            else:
                nelder_mead()
            iteration += 1

        best_index = np.argmin(fitness)
        return population[best_index]