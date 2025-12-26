import numpy as np

class MultiAgentEnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.crossover_rate = 0.9
        self.scaling_factor = 0.8
        self.temperature = 100.0
        self.cooling_rate = 0.993
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.7
        self.agent_count = 5  # Number of agents to create cooperative behavior

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def dynamic_neighborhood_search(self, position, global_best, bounds):
        neighbor_size = 0.1 * (bounds[:, 1] - bounds[:, 0])
        neighbor = position + np.random.uniform(-neighbor_size, neighbor_size, self.dim)
        return np.clip(neighbor, bounds[:, 0], bounds[:, 1])

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(population)
        personal_best = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        eval_count = self.population_size

        while eval_count < self.budget:
            mutation_prob = max(0.1, min(0.3, 1.0 - (eval_count / self.budget)))
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best[i] - population[i])
                    + self.social_constant * r2 * (global_best - population[i])
                )

                population[i] += velocities[i]
                levy_step = self.levy_flight(0.1 + 0.1 * (eval_count / self.budget))
                if np.random.rand() < mutation_prob:
                    population[i] += levy_step

                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])

                if eval_count + self.agent_count <= self.budget:
                    for _ in range(self.agent_count):
                        neighbor = self.dynamic_neighborhood_search(population[i], global_best, bounds)
                        fitness = func(neighbor)
                        eval_count += 1
                        if fitness < personal_best_fitness[i]:
                            personal_best[i] = neighbor
                            personal_best_fitness[i] = fitness
                            if fitness < global_best_fitness:
                                global_best = neighbor
                                global_best_fitness = fitness
                                self.cooling_rate *= self.cooling_rate

        return global_best, global_best_fitness