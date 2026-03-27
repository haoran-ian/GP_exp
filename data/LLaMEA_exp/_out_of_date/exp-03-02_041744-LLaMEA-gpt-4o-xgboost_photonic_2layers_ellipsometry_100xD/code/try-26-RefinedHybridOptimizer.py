import numpy as np

class RefinedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        
    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        pop_size = max(10, 5 * self.dim)
        de_cr = 0.9
        simplex_size = self.dim + 1
        restart_interval = self.budget // 10

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
                    de_f_dynamic = 0.4 + 0.6 * np.random.rand()
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

        def stochastic_simplex():
            nonlocal population, fitness
            for _ in range(self.budget - self.evaluations):
                if self.evaluations >= self.budget:
                    break
                simplex_indices = np.argsort(fitness)[:simplex_size]
                simplex = population[simplex_indices]
                simplex_fitness = fitness[simplex_indices]

                centroid = np.mean(simplex[:-1], axis=0)
                direction = np.random.randn(self.dim)
                direction /= np.linalg.norm(direction)
                reflection = np.clip(centroid + direction * np.random.uniform(0.5, 1.5), bounds[0], bounds[1])
                reflection_fitness = func(reflection)
                self.evaluations += 1

                if reflection_fitness < simplex_fitness[-1]:
                    simplex[-1] = reflection
                    simplex_fitness[-1] = reflection_fitness

                population[simplex_indices] = simplex
                fitness[simplex_indices] = simplex_fitness

        def random_restart():
            nonlocal population, fitness
            population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += pop_size

        iteration = 0
        while self.evaluations < self.budget:
            if iteration % restart_interval == 0 and iteration != 0:
                random_restart()
            elif iteration % 2 == 0:
                differential_evolution()
            else:
                stochastic_simplex()
            iteration += 1

        best_index = np.argmin(fitness)
        return population[best_index]