import numpy as np

class Enhanced_MultiSwarm_Adaptive_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.num_swarms = 5
        self.swarm_size = self.population_size // self.num_swarms
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.6
        self.social_coeff = 1.5
        self.temperature = 100.0
        self.cooling_rate = 0.98
        self.mutation_rate = 0.15
        self.current_evals = 0

        # Initialize particles for each swarm
        self.swarms = [np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        self.velocities = [np.random.uniform(-1, 1, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        self.personal_best_positions = [np.copy(swarm) for swarm in self.swarms]
        self.personal_best_scores = [np.full(self.swarm_size, float('inf')) for _ in range(self.num_swarms)]
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')
        self.elite_positions = [np.copy(swarm[:3]) for swarm in self.swarms]

    def __call__(self, func):
        while self.current_evals < self.budget:
            dynamic_inertia_weight = 0.4 + 0.5 * (1 - self.current_evals / self.budget)
            adaptive_cooling_rate = self.cooling_rate + 0.02 * np.sin(3 * np.pi * self.current_evals / self.budget)
            
            for s in range(self.num_swarms):
                for i in range(self.swarm_size):
                    score = func(self.swarms[s][i])
                    self.current_evals += 1
                    if score < self.personal_best_scores[s][i]:
                        self.personal_best_scores[s][i] = score
                        self.personal_best_positions[s][i] = self.swarms[s][i].copy()
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.swarms[s][i].copy()

                    r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                    cognitive_velocity = self.cognitive_coeff * r1 * (self.personal_best_positions[s][i] - self.swarms[s][i])
                    social_velocity = self.social_coeff * r2 * (self.global_best_position - self.swarms[s][i])
                    diversity_control = 0.5 * r3 * (self.elite_positions[s][np.random.randint(0, 3)] - self.swarms[s][i])
                    momentum = 0.1 * self.velocities[s][i]
                    self.velocities[s][i] = (dynamic_inertia_weight * self.velocities[s][i] +
                                              cognitive_velocity + social_velocity + diversity_control + momentum)
                    self.swarms[s][i] += self.velocities[s][i]

                    self.swarms[s][i] = np.clip(self.swarms[s][i], self.lower_bound, self.upper_bound)

                    dynamic_mutation_rate = self.mutation_rate * (1 - self.current_evals / self.budget)
                    if np.random.rand() < dynamic_mutation_rate:
                        mutation_vector = np.random.normal(0, 0.1, self.dim)
                        self.swarms[s][i] += mutation_vector
                        self.swarms[s][i] = np.clip(self.swarms[s][i], self.lower_bound, self.upper_bound)

                for i in range(self.swarm_size):
                    if self.current_evals >= self.budget:
                        break
                    candidate_solution = self.swarms[s][i] + np.random.normal(0, 0.1, self.dim)
                    candidate_solution = np.clip(candidate_solution, self.lower_bound, self.upper_bound)
                    candidate_score = func(candidate_solution)
                    self.current_evals += 1
                    if candidate_score < self.personal_best_scores[s][i] or \
                       np.exp((self.personal_best_scores[s][i] - candidate_score) / self.temperature) > np.random.rand():
                        self.swarms[s][i] = candidate_solution
                        self.personal_best_scores[s][i] = candidate_score
                        if candidate_score < self.global_best_score:
                            self.global_best_score = candidate_score
                            self.global_best_position = candidate_solution

            for s in range(self.num_swarms):
                if self.current_evals % (self.swarm_size * 2) == 0:
                    np.random.shuffle(self.elite_positions[s])

                sorted_indices = np.argsort(self.personal_best_scores[s])
                self.elite_positions[s] = self.personal_best_positions[s][sorted_indices[:3]]

            self.temperature *= adaptive_cooling_rate * (1 + 0.1 * np.cos(np.pi * self.current_evals / self.budget))

            self.cognitive_coeff = 1.5 + 0.2 * np.sin(2 * np.pi * self.current_evals / self.budget)
            self.social_coeff = 1.4 + 0.2 * np.sin(2 * np.pi * self.current_evals / self.budget)

        return self.global_best_position, self.global_best_score