import numpy as np

class AdaptiveMemorySwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, max(10, dim * 2))
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.fitness_function_calls = 0
        self.velocity_clamp = 0.1 * (self.upper_bound - self.lower_bound)
        self.memory = []  # Memory to store historically best positions

    def __call__(self, func):
        c1_initial, c2_initial = 2.5, 2.5
        c1_final, c2_final = 0.5, 0.5
        while self.fitness_function_calls < self.budget:
            for i in range(self.population_size):
                if self.fitness_function_calls >= self.budget:
                    break

                fitness = func(self.particles[i])
                self.fitness_function_calls += 1

                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.particles[i]

                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.particles[i]

            progress_ratio = self.fitness_function_calls / self.budget
            inertia_weight = 0.4 + 0.2 * np.random.rand() * (1 - progress_ratio)

            cognitive_constant = c1_initial - (c1_initial - c1_final) * progress_ratio
            social_constant = c2_initial - (c2_initial - c2_final) * progress_ratio

            # Dynamic neighborhood adjustment based on proximity to the global best
            neighborhood_radius = np.max([0.5 * (1 - progress_ratio), 0.1])
            
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Identify neighborhood of particles
                neighbors = np.where(np.linalg.norm(self.particles - self.particles[i], axis=1) < neighborhood_radius)[0]
                neighborhood_best = np.min(self.personal_best_scores[neighbors])
                
                self.velocities[i] *= inertia_weight
                self.velocities[i] += cognitive_constant * r1 * (self.personal_best_positions[i] - self.particles[i])
                self.velocities[i] += social_constant * r2 * (self.global_best_position - self.particles[i])
                if neighborhood_best < self.global_best_score:
                    self.velocities[i] += social_constant * r2 * (self.personal_best_positions[neighbors[np.argmin(self.personal_best_scores[neighbors])]] - self.particles[i])
                self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            # Store best positions in memory and use memory for diversification
            if len(self.memory) < self.population_size:
                self.memory.append(self.global_best_position)
            else:
                self.memory[np.random.randint(self.population_size)] = self.global_best_position

            opposition_particles = self.lower_bound + self.upper_bound - self.particles
            for i in range(self.population_size):
                if self.fitness_function_calls >= self.budget:
                    break

                opp_fitness = func(opposition_particles[i])
                self.fitness_function_calls += 1

                if opp_fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = opp_fitness
                    self.personal_best_positions[i] = opposition_particles[i]

                    if opp_fitness < self.global_best_score:
                        self.global_best_score = opp_fitness
                        self.global_best_position = opposition_particles[i]

            # Use memory positions to update elite particles
            elite_idxs = np.random.choice(np.argsort(self.personal_best_scores)[:max(1, self.population_size // 10)], max(1, self.population_size // 20), replace=False)
            for idx in elite_idxs:
                if np.random.rand() < 0.1:  # Small chance to diversify using memory
                    self.particles[idx] = self.memory[np.random.choice(len(self.memory))]
                else:
                    self.particles[idx] = self.personal_best_positions[idx]

        return self.global_best_position