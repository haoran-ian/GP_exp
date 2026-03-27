import numpy as np

class AdaptiveMemoryChaoticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.memory = np.full((5, dim), np.nan)  # Memory to store best solutions for adaptive crossover

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        pop_size = max(10, 5 * self.dim)
        simplex_size = self.dim + 1

        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += pop_size

        def update_memory(solution):
            # Store the best solutions encountered in memory
            worst_idx = np.argmax([func(mem) if not np.isnan(mem).any() else np.inf for mem in self.memory])
            if np.isnan(self.memory[worst_idx]).any() or func(solution) < func(self.memory[worst_idx]):
                self.memory[worst_idx] = solution

        def differential_evolution():
            nonlocal population, fitness

            for _ in range(self.budget - self.evaluations):
                if self.evaluations >= self.budget:
                    break
                for i in range(pop_size):
                    indices = np.random.choice(pop_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    de_f_dynamic = 0.5 + 0.5 * np.random.rand()
                    mutant = np.clip(x0 + de_f_dynamic * (x1 - x2), bounds[0], bounds[1])

                    if np.random.rand() < 0.5 and not np.isnan(self.memory).all():
                        # Use adaptive memory-based crossover
                        mem_idx = np.random.choice(np.where(~np.isnan(self.memory[:, 0]))[0])
                        cross_points = np.random.rand(self.dim) < 0.5 + 0.5 * np.random.rand()
                        trial = np.where(cross_points, self.memory[mem_idx], mutant)
                    else:
                        cross_points = np.random.rand(self.dim) < 0.9
                        if not np.any(cross_points):
                            cross_points[np.random.randint(0, self.dim)] = True
                        trial = np.where(cross_points, mutant, population[i])

                    trial = np.clip(trial, bounds[0], bounds[1])
                    f_trial = func(trial)
                    self.evaluations += 1
                    if f_trial < fitness[i]:
                        population[i] = trial
                        fitness[i] = f_trial
                        update_memory(trial)

        def chaotic_search():
            nonlocal population, fitness
            beta = 0.5 * (1 + np.sin(3 * np.pi * np.random.rand()))

            for _ in range(self.budget - self.evaluations):
                if self.evaluations >= self.budget:
                    break
                simplex_indices = np.argsort(fitness)[:simplex_size]
                simplex = population[simplex_indices]
                simplex_fitness = fitness[simplex_indices]

                centroid = np.mean(simplex[:-1], axis=0)
                chaotic_vector = centroid + beta * (simplex[-1] - centroid)
                chaotic_vector = np.clip(chaotic_vector, bounds[0], bounds[1])
                chaotic_fitness = func(chaotic_vector)
                self.evaluations += 1

                if chaotic_fitness < simplex_fitness[-1]:
                    simplex[-1] = chaotic_vector
                    simplex_fitness[-1] = chaotic_fitness

                population[simplex_indices] = simplex
                fitness[simplex_indices] = simplex_fitness
                update_memory(simplex[np.argmin(simplex_fitness)])

        iteration = 0
        while self.evaluations < self.budget:
            if iteration % 2 == 0:
                differential_evolution()
            else:
                chaotic_search()
            iteration += 1

        best_index = np.argmin(fitness)
        return population[best_index]