import numpy as np

class AdaptiveMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = min(100, budget // 10)
        self.inertia_weight = 0.7
        self.elite_archive = []
        self.num_swarms = 3
        self.phase_lengths = [self.budget // self.num_swarms] * self.num_swarms

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        def initialize_population():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def evaluate_population(population):
            fitness = np.array([func(ind) for ind in population])
            self.evaluations += len(population)
            return fitness

        def exploitation_step(best_individuals, velocities):
            for i in range(self.num_swarms):
                velocities[i] = self.inertia_weight * velocities[i] + np.random.normal(0, 0.1, (self.population_size, self.dim))
                neighbors = best_individuals[i] + velocities[i] * (np.random.rand(self.population_size, self.dim))
                neighbors = np.clip(neighbors, lb, ub)
            return neighbors, velocities

        def exploration_step():
            return np.random.uniform(lb, ub, (self.population_size, self.dim))

        def crossover(parent1, parent2):
            alpha = np.random.rand(self.dim)
            return alpha * parent1 + (1 - alpha) * parent2

        swarms = [initialize_population() for _ in range(self.num_swarms)]
        swarm_fitness = [evaluate_population(swarm) for swarm in swarms]
        velocities = [np.zeros((self.population_size, self.dim)) for _ in range(self.num_swarms)]

        for phase in range(self.num_swarms):
            if self.evaluations >= self.budget:
                break

            if phase == 0:
                # Global exploration phase
                swarms = [exploration_step() for _ in range(self.num_swarms)]
            elif phase == 1:
                # Intermediate phase with crossover and diversity preservation
                for _ in range(self.phase_lengths[phase]):
                    if self.evaluations >= self.budget:
                        break
                    for i in range(self.num_swarms):
                        parents = swarms[i][np.argsort(swarm_fitness[i])[:2]]
                        offspring = crossover(parents[0], parents[1])
                        offspring = np.clip(offspring, lb, ub)
                        offspring_fitness = func(offspring)
                        self.evaluations += 1
                        if offspring_fitness < max(swarm_fitness[i]):
                            replace_idx = np.argmax(swarm_fitness[i])
                            swarms[i][replace_idx] = offspring
                            swarm_fitness[i][replace_idx] = offspring_fitness
            else:
                # Local exploitation phase with directional mutation
                best_indices = [np.argmin(swarm_fitness[i]) for i in range(self.num_swarms)]
                best_individuals = [swarms[i][best_indices[i]] for i in range(self.num_swarms)]
                swarms, velocities = exploitation_step(best_individuals, velocities)

            swarm_fitness = [evaluate_population(swarm) for swarm in swarms]
            for i in range(self.num_swarms):
                self.elite_archive.append(swarms[i][np.argmin(swarm_fitness[i])])

        best_idx = np.argmin([func(ind) for ind in self.elite_archive])
        return self.elite_archive[best_idx]