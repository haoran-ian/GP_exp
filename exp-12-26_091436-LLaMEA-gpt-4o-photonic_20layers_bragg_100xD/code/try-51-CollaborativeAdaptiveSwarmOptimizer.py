import numpy as np

class CollaborativeAdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.9
        self.inertia_min = 0.4
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.inter_swarm_communication_rate = 0.05
        self.num_swarms = 3
        self.swarms_population = None
        self.swarms_velocities = None
        self.swarms_best_positions = None
        self.swarms_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf

    def initialize_swarms(self, lb, ub):
        self.swarms_population = [np.random.uniform(lb, ub, (self.population_size // self.num_swarms, self.dim))
                                  for _ in range(self.num_swarms)]
        self.swarms_velocities = [np.random.uniform(-0.1, 0.1, (self.population_size // self.num_swarms, self.dim))
                                  for _ in range(self.num_swarms)]
        self.swarms_best_positions = [np.copy(swarm) for swarm in self.swarms_population]
        self.swarms_best_scores = [np.full(self.population_size // self.num_swarms, np.inf) for _ in range(self.num_swarms)]
        self.global_best_position = np.random.uniform(lb, ub, self.dim)
        self.vel_clamp = 0.1 * (ub - lb)

    def update_particles(self, swarm_index, lb, ub):
        r1 = np.random.rand(self.population_size // self.num_swarms, self.dim)
        r2 = np.random.rand(self.population_size // self.num_swarms, self.dim)

        cognitive_velocity = self.cognitive_coefficient * r1 * (self.swarms_best_positions[swarm_index] - self.swarms_population[swarm_index])
        social_velocity = self.social_coefficient * r2 * (self.global_best_position - self.swarms_population[swarm_index])

        velocities = self.inertia_weight * self.swarms_velocities[swarm_index] + cognitive_velocity + social_velocity
        self.swarms_velocities[swarm_index] = np.clip(velocities, -self.vel_clamp, self.vel_clamp)

        self.swarms_population[swarm_index] += self.swarms_velocities[swarm_index]
        # Occasionally allow inter-swarm communication
        if np.random.rand() < self.inter_swarm_communication_rate:
            for i in range(self.num_swarms):
                if i != swarm_index:
                    other_swarm_best = self.swarms_best_positions[i][np.argmin(self.swarms_best_scores[i])]
                    self.swarms_population[swarm_index] = np.where(np.random.rand(self.population_size // self.num_swarms, self.dim) < 0.1,
                                                                   other_swarm_best,
                                                                   self.swarms_population[swarm_index])

    def evaluate_population(self, func, swarm_index):
        for i in range(self.population_size // self.num_swarms):
            score = func(self.swarms_population[swarm_index][i])
            if score < self.swarms_best_scores[swarm_index][i]:
                self.swarms_best_scores[swarm_index][i] = score
                self.swarms_best_positions[swarm_index][i] = self.swarms_population[swarm_index][i]
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.swarms_population[swarm_index][i]

    def adapt_parameters(self, progress):
        self.inertia_weight = self.inertia_min + (0.9 - self.inertia_min) * (1 - progress)
        self.cognitive_coefficient = 2.0 - 1.5 * progress
        self.social_coefficient = 1.0 + 1.5 * progress
        # Adapt velocity clamping dynamically
        self.vel_clamp = (0.1 + 0.1 * progress) * (ub - lb)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarms(lb, ub)

        evaluations = 0
        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.adapt_parameters(progress)

            for swarm_index in range(self.num_swarms):
                self.update_particles(swarm_index, lb, ub)
                self.evaluate_population(func, swarm_index)
                evaluations += self.population_size // self.num_swarms

        return self.global_best_position, self.global_best_score