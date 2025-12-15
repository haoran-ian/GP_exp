import numpy as np

class EnhancedAdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.population_size = self.initial_population_size
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = None
        self.pbest_positions = None
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
        self.mutation_probability = 0.1

    def __call__(self, func):
        bounds = func.bounds
        self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.pbest_positions = self.positions.copy()

        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                score = func(self.positions[i])
                eval_count += 1
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i].copy()

                r1, r2 = np.random.random(), np.random.random()
                # Nonlinear dynamic cognitive coefficient
                self.cognitive_coeff = 1.5 * np.exp(-0.005 * eval_count/self.budget)
                cognitive_velocity = self.cognitive_coeff * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = self.social_coeff * r2 * (self.gbest_position - self.positions[i])
                
                # Exponential decay for inertia weight
                self.inertia_weight = 0.9 * np.exp(-0.005 * eval_count/self.budget)
                
                self.velocities[i] = (self.inertia_weight * self.velocities[i] 
                                      + cognitive_velocity + social_velocity)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], bounds.lb, bounds.ub)

                # Mutation to maintain diversity
                if np.random.random() < self.mutation_probability:
                    mutation = np.random.normal(0, 0.1, self.dim)
                    self.positions[i] = np.clip(self.positions[i] + mutation, bounds.lb, bounds.ub)

            # Dynamic population resizing strategy
            self.population_size = int(self.initial_population_size * (1 - eval_count/self.budget))
            self.population_size = max(1, self.population_size)  # Ensure at least one individual

            self.positions = self.positions[:self.population_size]
            self.velocities = self.velocities[:self.population_size]
            self.pbest_positions = self.pbest_positions[:self.population_size]
            self.pbest_scores = self.pbest_scores[:self.population_size]

            if eval_count >= self.budget:
                break

        return self.gbest_position, self.gbest_score