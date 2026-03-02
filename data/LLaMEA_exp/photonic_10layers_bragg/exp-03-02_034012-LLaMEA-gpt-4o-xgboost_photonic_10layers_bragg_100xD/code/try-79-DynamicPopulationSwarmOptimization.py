import numpy as np

class DynamicPopulationSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_size = 5
        self.swarm = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_values = None
        self.global_best = None
        self.global_best_value = np.inf

    def initialize(self, lb, ub):
        self.population_size = self.initial_population_size
        self.swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.personal_best = np.copy(self.swarm)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best = np.copy(self.swarm[0])

    def update_velocity(self, inertia, personal_coefficient, global_coefficient):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        cognitive = personal_coefficient * r1 * (self.personal_best - self.swarm)
        social = global_coefficient * r2 * (self.global_best - self.swarm)
        self.velocity = inertia * self.velocity + cognitive + social

    def update_position(self, lb, ub):
        self.swarm += self.velocity
        self.swarm = np.clip(self.swarm, lb, ub)

    def stochastic_hill_climb(self, particle, lb, ub, scale=0.1):
        perturbation = np.random.standard_normal(self.dim) * scale
        neighbor = np.clip(particle + perturbation, lb, ub)
        return neighbor

    def diversity_mutation(self, particle, lb, ub, mutation_rate=0.1):
        mutation_vector = np.random.uniform(lb, ub, self.dim)
        mask = np.random.rand(self.dim) < mutation_rate
        return np.where(mask, mutation_vector, particle)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            fitness = func(self.swarm[i])
            if fitness < self.personal_best_values[i]:
                self.personal_best_values[i] = fitness
                self.personal_best[i] = self.swarm[i].copy()
            if fitness < self.global_best_value:
                self.global_best_value = fitness
                self.global_best = self.swarm[i].copy()

    def adapt_population_size(self, evaluations, budget):
        ratio = evaluations / budget
        self.population_size = int(self.initial_population_size * (1 - ratio) + 10)  # Dynamic size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            inertia = 0.9 - 0.9 * (evaluations / self.budget)
            personal_coefficient = 1.5 + np.random.rand()
            global_coefficient = 1.5 + np.random.rand()

            self.update_velocity(inertia, personal_coefficient, global_coefficient)
            self.update_position(lb, ub)
            self.evaluate_population(func)
            
            elite_indices = np.argsort(self.personal_best_values)[:self.elite_size]
            elites = self.swarm[elite_indices]

            for i in range(self.population_size):
                neighbor = self.stochastic_hill_climb(self.swarm[i], lb, ub)
                neighbor_fitness = func(neighbor)
                evaluations += 1
                if neighbor_fitness < self.personal_best_values[i]:
                    self.swarm[i] = neighbor
                    self.personal_best_values[i] = neighbor_fitness
                    self.personal_best[i] = neighbor
                if neighbor_fitness < self.global_best_value:
                    self.global_best_value = neighbor_fitness
                    self.global_best = neighbor

                if evaluations % 10 == 0:
                    for elite in elites:
                        adaptive_scale = 0.05 * (1 - evaluations / self.budget)
                        enhanced_neighbor = self.stochastic_hill_climb(elite, lb, ub, scale=adaptive_scale)
                        enhanced_fitness = func(enhanced_neighbor)
                        evaluations += 1
                        if enhanced_fitness < self.global_best_value:
                            self.global_best_value = enhanced_fitness
                            self.global_best = enhanced_neighbor
                
                if evaluations % 20 == 0:
                    for j in range(self.population_size):
                        mutated_particle = self.diversity_mutation(self.swarm[j], lb, ub)
                        mutated_fitness = func(mutated_particle)
                        evaluations += 1
                        if mutated_fitness < self.personal_best_values[j]:
                            self.swarm[j] = mutated_particle
                            self.personal_best_values[j] = mutated_fitness
                            self.personal_best[j] = mutated_particle
                        if mutated_fitness < self.global_best_value:
                            self.global_best_value = mutated_fitness
                            self.global_best = mutated_particle

                if evaluations >= self.budget:
                    break

            self.adapt_population_size(evaluations, self.budget)

        return self.global_best_value, self.global_best