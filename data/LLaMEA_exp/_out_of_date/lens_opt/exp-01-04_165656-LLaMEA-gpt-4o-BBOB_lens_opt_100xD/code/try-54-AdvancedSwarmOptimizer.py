import numpy as np

class AdvancedSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20 * dim
        self.learning_rate = 0.1
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4

    def local_search(self, individual, lb, ub):
        step_size = 0.05 * (ub - lb)
        perturbation = np.random.normal(0, step_size, self.dim)
        return np.clip(individual + perturbation, lb, ub)

    def adaptive_inertia_weight(self, progress):
        return self.inertia_weight_initial - (self.inertia_weight_initial - self.inertia_weight_final) * progress

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        global_best = population[best_idx]
        global_best_fitness = fitness[best_idx]
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()

        evaluations = population_size

        while evaluations < self.budget:
            progress = evaluations / self.budget
            inertia_weight = self.adaptive_inertia_weight(progress)
            for i in range(population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocity[i] = (inertia_weight * velocity[i] +
                               self.learning_rate * r1 * (personal_best[i] - population[i]) +
                               self.learning_rate * r2 * (global_best - population[i]))
                population[i] = np.clip(population[i] + velocity[i], lb, ub)
                trial_fitness = func(population[i])
                evaluations += 1

                if trial_fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = trial_fitness

                if trial_fitness < global_best_fitness:
                    global_best = population[i]
                    global_best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            # Diversity preservation mechanism
            diversity = np.mean(np.std(population, axis=0))
            if diversity < 0.1:  # Threshold can be tuned
                idx = np.random.choice(population_size)
                population[idx] = np.random.uniform(lb, ub, self.dim)

        return global_best