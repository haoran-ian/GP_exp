import numpy as np

class AdaptiveMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, max(10, dim * 2))
        self.num_swarms = 3
        self.swarm_size = self.population_size // self.num_swarms
        self.particles = [np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, dim)) for _ in range(self.num_swarms)]
        self.velocities = [np.random.uniform(-1, 1, (self.swarm_size, dim)) for _ in range(self.num_swarms)]
        self.personal_best_positions = [np.copy(particles) for particles in self.particles]
        self.personal_best_scores = [np.full(self.swarm_size, np.inf) for _ in range(self.num_swarms)]
        self.global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, dim)
        self.global_best_score = np.inf
        self.fitness_function_calls = 0
        self.velocity_clamp = 0.1 * (self.upper_bound - self.lower_bound)

    def __call__(self, func):
        c1_initial, c2_initial = 2.5, 2.5
        c1_final, c2_final = 0.5, 0.5
        while self.fitness_function_calls < self.budget:
            for s in range(self.num_swarms):
                for i in range(self.swarm_size):
                    if self.fitness_function_calls >= self.budget:
                        break

                    fitness = func(self.particles[s][i])
                    self.fitness_function_calls += 1

                    if fitness < self.personal_best_scores[s][i]:
                        self.personal_best_scores[s][i] = fitness
                        self.personal_best_positions[s][i] = self.particles[s][i]

                    if fitness < self.global_best_score:
                        self.global_best_score = fitness
                        self.global_best_position = self.particles[s][i]

            progress_ratio = self.fitness_function_calls / self.budget
            inertia_weight = 0.4 + 0.2 * np.random.rand() * (1 - progress_ratio)

            for s in range(self.num_swarms):
                cognitive_constant = c1_initial - (c1_initial - c1_final) * progress_ratio
                social_constant = c2_initial - (c2_initial - c2_final) * progress_ratio

                for i in range(self.swarm_size):
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    self.velocities[s][i] *= inertia_weight
                    self.velocities[s][i] += cognitive_constant * r1 * (self.personal_best_positions[s][i] - self.particles[s][i])
                    self.velocities[s][i] += social_constant * r2 * (self.global_best_position - self.particles[s][i])
                    self.velocities[s][i] = np.clip(self.velocities[s][i], -self.velocity_clamp, self.velocity_clamp)
                    self.particles[s][i] += self.velocities[s][i]
                    self.particles[s][i] = np.clip(self.particles[s][i], self.lower_bound, self.upper_bound)

                # Perform competition and cooperation between swarms
                if s > 0:
                    for i in range(self.swarm_size):
                        if np.random.rand() < 0.1:
                            self.particles[s][i] = np.copy(self.particles[s-1][np.random.randint(self.swarm_size)])

            # Update global best from all swarms
            for s in range(self.num_swarms):
                best_idx = np.argmin(self.personal_best_scores[s])
                if self.personal_best_scores[s][best_idx] < self.global_best_score:
                    self.global_best_score = self.personal_best_scores[s][best_idx]
                    self.global_best_position = np.copy(self.personal_best_positions[s][best_idx])

        return self.global_best_position