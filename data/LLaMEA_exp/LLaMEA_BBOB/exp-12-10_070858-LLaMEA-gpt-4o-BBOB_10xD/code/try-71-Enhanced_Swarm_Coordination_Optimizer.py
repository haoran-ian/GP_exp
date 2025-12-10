import numpy as np

class Enhanced_Swarm_Coordination_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_count = 3
        self.population_size = 20
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.6
        self.social_coeff = 1.5
        self.temperature = 100.0
        self.cooling_rate = 0.98
        self.mutation_rate = 0.15
        self.current_evals = 0

        # Initialize swarms
        self.swarms = [np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim)) for _ in range(self.swarm_count)]
        self.velocities = [np.random.uniform(-1, 1, (self.population_size, self.dim)) for _ in range(self.swarm_count)]
        self.personal_best_positions = [np.copy(swarm) for swarm in self.swarms]
        self.personal_best_scores = [np.full(self.population_size, float('inf')) for _ in range(self.swarm_count)]
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')

    def __call__(self, func):
        while self.current_evals < self.budget:
            for swarm_index in range(self.swarm_count):
                swarm = self.swarms[swarm_index]
                for i in range(self.population_size):
                    score = func(swarm[i])
                    self.current_evals += 1
                    if score < self.personal_best_scores[swarm_index][i]:
                        self.personal_best_scores[swarm_index][i] = score
                        self.personal_best_positions[swarm_index][i] = swarm[i].copy()
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = swarm[i].copy()

                    r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                    cognitive_velocity = self.cognitive_coeff * r1 * (self.personal_best_positions[swarm_index][i] - swarm[i])
                    social_velocity = self.social_coeff * r2 * (self.global_best_position - swarm[i])
                    diversity_control = 0.5 * r3 * (np.mean(self.swarms, axis=0)[i] - swarm[i])
                    momentum = 0.1 * self.velocities[swarm_index][i]
                    self.velocities[swarm_index][i] = (self.inertia_weight * self.velocities[swarm_index][i] +
                                                       cognitive_velocity + social_velocity + diversity_control + momentum)
                    swarm[i] += self.velocities[swarm_index][i]
                    swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

                    dynamic_mutation_rate = self.mutation_rate * (1 - self.current_evals / self.budget)
                    if np.random.rand() < dynamic_mutation_rate:
                        mutation_vector = np.random.normal(0, 0.1 + 0.1 * (np.sin(np.pi * self.current_evals / self.budget)), self.dim)
                        swarm[i] += mutation_vector
                        swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

                # Inter-swarm communication
                if swarm_index < self.swarm_count - 1:
                    self.swarms[swarm_index + 1][:] = 0.9 * self.swarms[swarm_index + 1][:] + 0.1 * self.swarms[swarm_index][:]
                    
            # Adaptive parameter adjustments
            self.inertia_weight = 0.4 + 0.5 * (1 - self.current_evals / self.budget)
            self.cognitive_coeff = 1.5 + 0.2 * np.sin(2 * np.pi * self.current_evals / self.budget)
            self.social_coeff = 1.4 + 0.2 * np.sin(2 * np.pi * self.current_evals / self.budget)
            self.temperature *= self.cooling_rate * (1 + 0.1 * np.cos(np.pi * self.current_evals / self.budget))

        return self.global_best_position, self.global_best_score