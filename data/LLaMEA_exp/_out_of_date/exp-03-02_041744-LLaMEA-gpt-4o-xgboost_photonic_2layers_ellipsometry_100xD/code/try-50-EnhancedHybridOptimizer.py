import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        
    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        pop_size = max(10, 5 * self.dim)  # Adjusted population size based on problem size
        de_cr = 0.9  # DE crossover probability
        simplex_size = self.dim + 1  # Size of Nelder-Mead simplex

        # Initialize population for DE
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += pop_size

        def differential_evolution():
            nonlocal population, fitness
            
            for _ in range(self.budget - self.evaluations):
                if self.evaluations >= self.budget:
                    break
                for i in range(pop_size):
                    indices = np.random.choice(pop_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    de_f_dynamic = 0.5 + 0.5 * np.random.rand()  # Dynamic adaptation of mutation factor
                    mutant = np.clip(x0 + de_f_dynamic * (x1 - x2), bounds[0], bounds[1])
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
                    expansion_factor = 1.5 if np.random.rand() > 0.5 else 2  # Adaptive factor
                    expansion = np.clip(centroid + expansion_factor * (reflection - centroid), bounds[0], bounds[1])
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
                    else:  # Shrink the simplex
                        simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                        simplex_fitness[1:] = np.array([func(simplex[i]) for i in range(1, simplex_size)])
                        self.evaluations += simplex_size - 1

                population[simplex_indices] = simplex
                fitness[simplex_indices] = simplex_fitness

        # Hybrid strategy
        iteration = 0
        while self.evaluations < self.budget:
            if iteration % 2 == 0:
                differential_evolution()
            else:
                nelder_mead()
            iteration += 1

        best_index = np.argmin(fitness)
        return population[best_index]