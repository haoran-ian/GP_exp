import numpy as np

class EnhancedSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)

        # Initialize parameters for Particle Swarm Optimization Enhanced with Differential Evolution strategies
        population_size = 20 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T

        # Initialize population
        population = np.random.rand(population_size, self.dim)
        population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        velocity = np.zeros((population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        # Initialize personal best and global best
        p_best = population.copy()
        p_best_fitness = fitness.copy()
        g_best_idx = np.argmin(fitness)
        g_best = population[g_best_idx].copy()

        # Adaptive parameters
        inertia_weight = 0.7
        cognitive_coeff = 1.5
        social_coeff = 1.5

        # Memory for adaptive DE
        success_mem = []
        archive = []

        while eval_count < self.budget:
            for i in range(population_size):
                # Update velocity
                inertia = inertia_weight * velocity[i]
                cognitive = cognitive_coeff * np.random.rand(self.dim) * (p_best[i] - population[i])
                social = social_coeff * np.random.rand(self.dim) * (g_best - population[i])
                velocity[i] = inertia + cognitive + social

                # Update position
                population[i] += velocity[i]
                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])
                
                # Evaluate particle
                f_value = func(population[i])
                eval_count += 1

                # Update personal best
                if f_value < p_best_fitness[i]:
                    p_best[i] = population[i].copy()
                    p_best_fitness[i] = f_value

                    # Adaptive DE based on success history
                    if success_mem:
                        F = np.mean(success_mem)
                        F = np.clip(F + np.random.normal(0, 0.05), 0.5, 1.2)
                    else:
                        F = 0.7 + np.random.uniform(-0.1, 0.3)

                    CR = 0.9 + np.random.uniform(-0.1, 0.1)

                    # Mutation and crossover (DE variation)
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                    f_trial = func(trial)
                    eval_count += 1

                    if f_trial < f_value:
                        population[i] = trial
                        fitness[i] = f_trial
                        success_mem.append(F)
                        if len(success_mem) > 10:
                            success_mem.pop(0)
                        archive.append(trial)

                # Update global best
                if f_value < fitness[g_best_idx]:
                    g_best_idx = i
                    g_best = population[i].copy()

                # Archive-assisted exploration
                if eval_count < self.budget and np.random.rand() < 0.3 and len(archive) > 0:
                    archive_idx = np.random.randint(len(archive))
                    perturbed = archive[archive_idx] + np.random.normal(0, 0.1, self.dim)
                    perturbed = np.clip(perturbed, bounds[:, 0], bounds[:, 1])
                    f_perturbed = func(perturbed)
                    eval_count += 1
                    if f_perturbed < fitness[g_best_idx]:
                        g_best = perturbed
                        g_best_idx = np.argmin([f_perturbed, fitness[g_best_idx]])

        return g_best