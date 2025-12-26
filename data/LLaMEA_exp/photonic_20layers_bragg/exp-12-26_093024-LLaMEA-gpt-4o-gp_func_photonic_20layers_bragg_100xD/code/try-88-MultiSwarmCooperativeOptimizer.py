import numpy as np

class MultiSwarmCooperativeOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        num_swarms = 3
        swarm_size = 5
        population_size = num_swarms * swarm_size
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # Adaptive parameters
        F_base = 0.8
        CR_base = 0.9
        diversity_threshold = 0.1

        def calculate_diversity(swarm):
            centroid = np.mean(swarm, axis=0)
            diversity = np.mean(np.linalg.norm(swarm - centroid, axis=1))
            return diversity

        while evaluations < self.budget:
            # Iterate over each swarm
            for swarm_idx in range(num_swarms):
                start, end = swarm_idx * swarm_size, (swarm_idx + 1) * swarm_size
                swarm = population[start:end]
                swarm_fitness = fitness[start:end]

                best_idx_swarm = np.argmin(swarm_fitness)
                diversity = calculate_diversity(swarm)

                if diversity < diversity_threshold:
                    F = F_base * (1 + np.random.rand() * 0.5)
                    CR = CR_base * (1 - np.random.rand() * 0.5)
                else:
                    F = F_base
                    CR = CR_base

                # Perform DE for the swarm
                for i in range(swarm_size):
                    if evaluations >= self.budget:
                        break

                    idxs = [idx for idx in range(swarm_size) if idx != i]
                    a, b, c = swarm[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)

                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True

                    trial = np.where(cross_points, mutant, swarm[i])
                    trial_fitness = func(trial)
                    evaluations += 1

                    if trial_fitness < swarm_fitness[i]:
                        swarm[i] = trial
                        swarm_fitness[i] = trial_fitness

                # Elitism and update global population
                best_idx_global = np.argmin(fitness)
                if evaluations < self.budget:
                    rand_ind = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                    rand_fitness = func(rand_ind)
                    evaluations += 1

                    if rand_fitness < fitness[np.argmax(fitness)]:
                        worst_idx = np.argmax(fitness)
                        fitness[worst_idx] = rand_fitness
                        population[worst_idx] = rand_ind

                population[start:end] = swarm
                fitness[start:end] = swarm_fitness
                population[best_idx_global] = population[np.argmin(fitness)]
                fitness[best_idx_global] = min(fitness)

        # Return best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]