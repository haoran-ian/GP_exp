import numpy as np

class AdaptiveMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.current_evals = 0
        self.mutation_rate = 0.1
        self.swarm_count = 3

        # Initialize swarms
        self.swarms = [self._initialize_swarm() for _ in range(self.swarm_count)]

    def _initialize_swarm(self):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, float('inf'))
        global_best_position = np.zeros(self.dim)
        global_best_score = float('inf')
        return {
            "particles": particles,
            "velocities": velocities,
            "personal_best_positions": personal_best_positions,
            "personal_best_scores": personal_best_scores,
            "global_best_position": global_best_position,
            "global_best_score": global_best_score
        }

    def __call__(self, func):
        while self.current_evals < self.budget:
            for swarm in self.swarms:
                particles = swarm["particles"]
                velocities = swarm["velocities"]
                personal_best_positions = swarm["personal_best_positions"]
                personal_best_scores = swarm["personal_best_scores"]
                global_best_position = swarm["global_best_position"]
                global_best_score = swarm["global_best_score"]

                for i in range(self.population_size):
                    # Evaluate fitness
                    score = func(particles[i])
                    self.current_evals += 1
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = particles[i].copy()
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[i].copy()

                    # Update velocity and position with adaptive learning rates
                    r1, r2 = np.random.rand(), np.random.rand()
                    learning_rate = 0.5 + (0.5 * self.current_evals / self.budget)
                    cognitive_velocity = learning_rate * self.cognitive_coeff * r1 * (personal_best_positions[i] - particles[i])
                    social_velocity = learning_rate * self.social_coeff * r2 * (global_best_position - particles[i])
                    velocities[i] = (self.inertia_weight * velocities[i] + cognitive_velocity + social_velocity)
                    particles[i] += velocities[i]

                    # Ensure particles are within bounds
                    particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                    # Apply mutation with dynamic rate
                    dynamic_mutation_rate = self.mutation_rate * (1 - self.current_evals / self.budget)
                    if np.random.rand() < dynamic_mutation_rate:
                        mutation_vector = np.random.normal(0, 1, self.dim)
                        particles[i] += mutation_vector
                        particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                # Update global best for the swarm
                swarm["global_best_position"] = global_best_position
                swarm["global_best_score"] = global_best_score

            # Swarm coordination
            best_swarm = min(self.swarms, key=lambda s: s["global_best_score"])
            for swarm in self.swarms:
                if np.random.rand() < 0.5:
                    swarm["global_best_position"] = best_swarm["global_best_position"]
                    swarm["global_best_score"] = best_swarm["global_best_score"]

        # Return the best position and score found
        best_swarm = min(self.swarms, key=lambda s: s["global_best_score"])
        return best_swarm["global_best_position"], best_swarm["global_best_score"]