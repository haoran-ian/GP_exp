import numpy as np

class EnhancedHybridAMES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(dim))
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia_weight = 0.9  # Start with higher inertia
        self.inertia_decay = 0.99  # Decay for adaptive inertia
        self.cognitive_coeff = 1.7
        self.social_coeff = 1.7
        self.adaptive_momentum = 0.5
        self.elite_rate = 0.15
        self.exploration_rate = 0.3
        self.de_scale = 0.8
        self.de_cross_prob = 0.9

    def initialize(self, bounds):
        self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, float('inf'))

    def chaotic_map(self, iteration, max_iterations):
        beta = 0.7  # Control parameter for logistic map
        x = 0.6  # Initial value for logistic map
        for _ in range(iteration):
            x = beta * x * (1 - x)
        return x

    def update_particles(self, func, iteration, max_iterations):
        self.elite_rate = 0.15 * (1 - iteration / max_iterations)  # Dynamic elite_rate
        elite_threshold = int(self.pop_size * self.elite_rate)
        for i in range(self.pop_size):
            score = func(self.positions[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i]
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[i]

        sorted_indices = np.argsort(self.personal_best_scores)
        elite_indices = sorted_indices[:elite_threshold]

        for i in range(self.pop_size):
            cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
            social_component = self.social_coeff * np.random.rand(self.dim) * (self.global_best_position - self.positions[i])
            elite_component = np.mean([self.positions[idx] for idx in elite_indices], axis=0) - self.positions[i]
            chaotic_factor = self.chaotic_map(iteration, max_iterations)

            self.velocities[i] = self.inertia_weight * self.velocities[i] \
                                 + cognitive_component \
                                 + social_component \
                                 + self.adaptive_momentum * elite_component \
                                 + chaotic_factor * self.velocities[i]

            momentum = self.exploration_rate * (np.random.rand(self.dim) - 0.5)
            self.velocities[i] += momentum

            self.positions[i] += self.velocities[i]
            self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

    def differential_evolution(self, func):
        trial_positions = np.copy(self.positions)
        for i in range(self.pop_size):
            indices = list(range(self.pop_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = self.positions[a] + self.de_scale * (self.positions[b] - self.positions[c])
            cross_points = np.random.rand(self.dim) < self.de_cross_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial_positions[i] = np.where(cross_points, mutant_vector, self.positions[i])
            trial_positions[i] = np.clip(trial_positions[i], func.bounds.lb, func.bounds.ub)
            trial_score = func(trial_positions[i])
            if trial_score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = trial_score
                self.positions[i] = trial_positions[i]
            if trial_score < self.global_best_score:
                self.global_best_score = trial_score
                self.global_best_position = trial_positions[i]

    def __call__(self, func):
        bounds = func.bounds
        self.initialize(bounds)
        evaluations = 0
        max_iterations = self.budget // self.pop_size
        while evaluations < self.budget:
            iteration = evaluations // self.pop_size
            self.update_particles(func, iteration, max_iterations)
            self.differential_evolution(func)
            self.inertia_weight *= self.inertia_decay  # Decay inertia weight
            evaluations += self.pop_size
        return self.global_best_position, self.global_best_score