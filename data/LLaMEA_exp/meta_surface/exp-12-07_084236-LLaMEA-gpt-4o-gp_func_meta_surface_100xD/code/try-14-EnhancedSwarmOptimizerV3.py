import numpy as np

class EnhancedSwarmOptimizerV3:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, max(10, dim * 2))
        self.num_swarms = 3  # Utilize multiple swarms for diversity
        self.swarms = [np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim)) for _ in range(self.num_swarms)]
        self.velocities = [np.random.uniform(-1, 1, (self.population_size, dim)) for _ in range(self.num_swarms)]
        self.personal_best_positions = [np.copy(swarm) for swarm in self.swarms]
        self.personal_best_scores = [np.full(self.population_size, np.inf) for _ in range(self.num_swarms)]
        self.global_best_position = None
        self.global_best_score = np.inf
        self.fitness_function_calls = 0
        self.velocity_clamp = 0.1 * (self.upper_bound - self.lower_bound)  # Adaptive velocity clamping
        
    def __call__(self, func):
        c1_initial, c2_initial = 2.5, 2.5
        c1_final, c2_final = 0.5, 0.5
        while self.fitness_function_calls < self.budget:
            for swarm_index in range(self.num_swarms):
                swarm, velocities, p_best_positions, p_best_scores = (
                    self.swarms[swarm_index],
                    self.velocities[swarm_index],
                    self.personal_best_positions[swarm_index],
                    self.personal_best_scores[swarm_index]
                )

                for i in range(self.population_size):
                    if self.fitness_function_calls >= self.budget:
                        break

                    fitness = func(swarm[i])
                    self.fitness_function_calls += 1

                    if fitness < p_best_scores[i]:
                        p_best_scores[i] = fitness
                        p_best_positions[i] = swarm[i]

                    if fitness < self.global_best_score:
                        self.global_best_score = fitness
                        self.global_best_position = swarm[i]

                progress_ratio = self.fitness_function_calls / self.budget
                inertia_weight = 0.4 + 0.3 * np.random.rand() * (1 - progress_ratio)  # Dynamic scaling of inertia

                cognitive_constant = c1_initial - (c1_initial - c1_final) * progress_ratio
                social_constant = c2_initial - (c2_initial - c2_final) * progress_ratio

                for i in range(self.population_size):
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    velocities[i] *= inertia_weight
                    velocities[i] += cognitive_constant * r1 * (p_best_positions[i] - swarm[i])
                    velocities[i] += social_constant * r2 * (self.global_best_position - swarm[i])
                    velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                    swarm[i] += velocities[i]
                    swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

                # Opposition-based learning for diversification
                opposition_particles = self.lower_bound + self.upper_bound - swarm
                for i in range(self.population_size):
                    if self.fitness_function_calls >= self.budget:
                        break

                    opp_fitness = func(opposition_particles[i])
                    self.fitness_function_calls += 1

                    if opp_fitness < p_best_scores[i]:
                        p_best_scores[i] = opp_fitness
                        p_best_positions[i] = opposition_particles[i]

                        if opp_fitness < self.global_best_score:
                            self.global_best_score = opp_fitness
                            self.global_best_position = opposition_particles[i]

            # Periodically regroup particles to promote synergy
            if self.fitness_function_calls % (self.budget // 10) == 0:
                best_swarm_index = np.argmin([np.min(p_best_scores) for p_best_scores in self.personal_best_scores])
                self.global_best_position = self.personal_best_positions[best_swarm_index][np.argmin(self.personal_best_scores[best_swarm_index])]
                self.global_best_score = np.min(self.personal_best_scores[best_swarm_index])

        return self.global_best_position