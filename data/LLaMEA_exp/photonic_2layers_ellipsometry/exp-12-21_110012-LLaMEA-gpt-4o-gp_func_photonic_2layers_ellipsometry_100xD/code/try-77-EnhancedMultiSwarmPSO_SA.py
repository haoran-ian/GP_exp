import numpy as np

class EnhancedMultiSwarmPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = max(2, dim // 10)
        self.num_particles_per_swarm = min(max(3, dim // 2), 20)
        self.w_min = 0.3
        self.w_max = 0.9
        self.c1_initial = 2.5
        self.c2_initial = 1.5
        self.c1_final = 0.5
        self.c2_final = 2.5
        self.temp_initial = 20
        self.temp_final = 0.1
        self.cooling_rate = 0.85
        self.positions = [None] * self.num_swarms
        self.velocities = [None] * self.num_swarms
        self.best_individual_positions = [None] * self.num_swarms
        self.best_individual_scores = [None] * self.num_swarms
        self.global_best_position = np.random.rand(self.dim)
        self.global_best_score = float('inf')
        self.function_evaluations = 0
        self.base_mutation_rate = 0.05
        self.inter_swarm_rate = np.linspace(0.1, 0.5, self.num_swarms)

    def __call__(self, func):
        np.random.seed(42)
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub

        for s in range(self.num_swarms):
            self.positions[s] = np.random.uniform(lb, ub, (self.num_particles_per_swarm, self.dim))
            self.velocities[s] = np.random.uniform(-1, 1, (self.num_particles_per_swarm, self.dim))
            self.best_individual_positions[s] = self.positions[s].copy()
            self.best_individual_scores[s] = np.array([func(position) for position in self.positions[s]])
            self.function_evaluations += self.num_particles_per_swarm

            swarm_best_idx = np.argmin(self.best_individual_scores[s])
            if self.best_individual_scores[s][swarm_best_idx] < self.global_best_score:
                self.global_best_score = self.best_individual_scores[s][swarm_best_idx]
                self.global_best_position = self.positions[s][swarm_best_idx].copy()

        while self.function_evaluations < self.budget:
            t = self.function_evaluations / self.budget
            c1 = self.c1_final + 0.5 * np.sin(2 * np.pi * t) * (self.c1_initial - self.c1_final)
            c2 = self.c2_final + 0.5 * np.cos(2 * np.pi * t) * (self.c2_initial - self.c2_final)
            self.w = self.w_max - ((self.w_max - self.w_min) * t)

            for s in range(self.num_swarms):
                # Calculate diversity
                diversity = np.mean(np.std(self.positions[s], axis=0))
                self.mutation_rate = self.base_mutation_rate + (0.1 * diversity)

                for i in range(self.num_particles_per_swarm):
                    r1, r2 = np.random.rand(), np.random.rand()
                    self.velocities[s][i] = (self.w * self.velocities[s][i] +
                                             c1 * r1 * (self.best_individual_positions[s][i] - self.positions[s][i]) +
                                             c2 * r2 * (self.global_best_position - self.positions[s][i]))
                    self.positions[s][i] = np.clip(self.positions[s][i] + self.velocities[s][i], lb, ub)

                    # Boundary reflection
                    self.positions[s][i] = np.where(self.positions[s][i] < lb, lb + (lb - self.positions[s][i]), self.positions[s][i])
                    self.positions[s][i] = np.where(self.positions[s][i] > ub, ub - (self.positions[s][i] - ub), self.positions[s][i])

                    score = func(self.positions[s][i])
                    self.function_evaluations += 1

                    if score < self.best_individual_scores[s][i]:
                        self.best_individual_scores[s][i] = score
                        self.best_individual_positions[s][i] = self.positions[s][i].copy()

                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.positions[s][i].copy()

                temp = self.temp_initial * ((self.cooling_rate + np.random.rand() * 0.05) ** (t))
                if temp > self.temp_final:
                    for i in range(self.num_particles_per_swarm):
                        candidate_position = self.positions[s][i] + np.random.normal(0, temp * (1 - t), self.dim)
                        candidate_position = np.clip(candidate_position, lb, ub)
                        candidate_score = func(candidate_position)
                        self.function_evaluations += 1

                        if candidate_score < self.best_individual_scores[s][i] or \
                           np.random.rand() < np.exp(-(candidate_score - self.best_individual_scores[s][i]) / temp):
                            self.positions[s][i] = candidate_position
                            self.best_individual_scores[s][i] = candidate_score
                            self.best_individual_positions[s][i] = candidate_position

                            if candidate_score < self.global_best_score:
                                self.global_best_score = candidate_score
                                self.global_best_position = candidate_position

                # Adaptive Mutation Strategy
                if np.random.rand() < self.mutation_rate:
                    mutation_percentage = 0.1 * (1 - np.exp(-5 * t))
                    mutation_vector = np.random.uniform(-mutation_percentage, mutation_percentage, self.dim)
                    for i in range(self.num_particles_per_swarm):
                        mutated_position = self.positions[s][i] + mutation_vector
                        mutated_position = np.clip(mutated_position, lb, ub)
                        mutated_score = func(mutated_position)
                        self.function_evaluations += 1

                        if mutated_score < self.best_individual_scores[s][i]:
                            self.best_individual_scores[s][i] = mutated_score
                            self.best_individual_positions[s][i] = mutated_position

                            if mutated_score < self.global_best_score:
                                self.global_best_score = mutated_score
                                self.global_best_position = mutated_position

                # Inter-swarm communication
                if np.random.rand() < self.inter_swarm_rate[s]:
                    for other_s in range(self.num_swarms):
                        if other_s != s:
                            external_best_idx = np.argmin(self.best_individual_scores[other_s])
                            external_best_position = self.positions[other_s][external_best_idx].copy()
                            self.velocities[s] = (self.velocities[s] + 
                                                  np.random.uniform(-0.5, 0.5, self.velocities[s].shape) *
                                                  (external_best_position - self.positions[s]))

        return self.global_best_position, self.global_best_score