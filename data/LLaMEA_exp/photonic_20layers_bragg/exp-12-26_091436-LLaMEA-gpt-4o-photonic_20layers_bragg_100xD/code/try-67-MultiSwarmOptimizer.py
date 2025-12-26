import numpy as np

class MultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.swarms = 3
        self.inertia_weight = 0.9
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.vel_clamp = None
        self.populations = [None] * self.swarms
        self.velocities = [None] * self.swarms
        self.personal_best_positions = [None] * self.swarms
        self.personal_best_scores = [None] * self.swarms
        self.global_best_position = None
        self.global_best_score = np.inf

    def initialize_population(self, lb, ub):
        for swarm in range(self.swarms):
            self.populations[swarm] = np.random.uniform(lb, ub, (self.population_size, self.dim))
            self.velocities[swarm] = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
            self.personal_best_positions[swarm] = np.copy(self.populations[swarm])
            self.personal_best_scores[swarm] = np.full(self.population_size, np.inf)
        self.global_best_position = np.random.uniform(lb, ub, self.dim)
        self.vel_clamp = 0.1 * (ub - lb)
    
    def update_particles(self, swarm):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)

        cognitive_velocity = self.cognitive_coefficient * r1 * (self.personal_best_positions[swarm] - self.populations[swarm])
        social_velocity = self.social_coefficient * r2 * (self.global_best_position - self.populations[swarm])

        # Regroup swarm particles randomly to maintain diversity
        if np.random.rand() < 0.1:
            shuffle_indices = np.random.permutation(self.population_size)
            self.populations[swarm] = self.populations[swarm][shuffle_indices]
            self.velocities[swarm] = self.velocities[swarm][shuffle_indices]

        self.velocities[swarm] = self.inertia_weight * self.velocities[swarm] + cognitive_velocity + social_velocity
        self.velocities[swarm] = np.clip(self.velocities[swarm], -self.vel_clamp, self.vel_clamp)

        self.populations[swarm] += self.velocities[swarm]
        # Randomly reinitialize a fraction of the particles
        if np.random.rand() < 0.05:
            random_indices = np.random.choice(self.population_size, size=int(0.1 * self.population_size), replace=False)
            self.populations[swarm][random_indices] = np.random.uniform(lb, ub, (len(random_indices), self.dim))

    def evaluate_population(self, func, swarm):
        for i in range(self.population_size):
            score = func(self.populations[swarm][i])
            if score < self.personal_best_scores[swarm][i]:
                self.personal_best_scores[swarm][i] = score
                self.personal_best_positions[swarm][i] = self.populations[swarm][i]
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.populations[swarm][i]

    def adapt_parameters(self, progress):
        self.inertia_weight = 0.9 - 0.5 * progress
        self.cognitive_coefficient = 2.0 - 1.5 * progress
        self.social_coefficient = 1.0 + 1.5 * progress
        # Dynamic velocity clamping
        self.vel_clamp = (0.1 + 0.1 * progress) * (ub - lb)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)

        evaluations = 0
        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.adapt_parameters(progress)
            for swarm in range(self.swarms):
                self.update_particles(swarm)
                self.evaluate_population(func, swarm)
                evaluations += self.population_size
                if evaluations >= self.budget:
                    break

        return self.global_best_position, self.global_best_score