import numpy as np

class EnhancedQuantumHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.9
        self.eval_count = 0
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.quantum_exponent = 0.01
        self.num_swarms = 3
        self.swarm_pop = self.population_size // self.num_swarms

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        swarms = [np.random.uniform(bounds[:, 0], bounds[:, 1], (self.swarm_pop, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.swarm_pop, self.dim)) for _ in range(self.num_swarms)]
        personal_best_positions = [np.copy(swarm) for swarm in swarms]
        personal_best_values = [np.array([func(p) for p in swarm]) for swarm in swarms]

        global_best_position = None
        global_best_value = float('inf')

        for swarm_idx in range(self.num_swarms):
            global_best_idx = np.argmin(personal_best_values[swarm_idx])
            candidate_best_position = personal_best_positions[swarm_idx][global_best_idx, :]
            candidate_best_value = personal_best_values[swarm_idx][global_best_idx]

            if candidate_best_value < global_best_value:
                global_best_position = candidate_best_position
                global_best_value = candidate_best_value

        def update_particle_velocity(velocity, particle, personal_best, global_best):
            r1, r2 = np.random.rand(2)
            dynamic_social_coefficient = self.social_coefficient * (1 - self.eval_count / self.budget)
            cognitive_velocity = self.cognitive_coefficient * r1 * (personal_best - particle)
            social_velocity = dynamic_social_coefficient * r2 * (global_best - particle)
            adaptive_scale = 1 / (1 + np.exp(-self.eval_count / (0.2 * self.budget)))
            new_velocity = (self.inertia_weight * velocity + cognitive_velocity + social_velocity) * adaptive_scale
            return new_velocity

        while self.eval_count < self.budget:
            for swarm_idx in range(self.num_swarms):
                swarm = swarms[swarm_idx]
                velocity = velocities[swarm_idx]
                personal_best_pos = personal_best_positions[swarm_idx]
                personal_best_vals = personal_best_values[swarm_idx]

                for i in range(self.swarm_pop):
                    velocity[i] = update_particle_velocity(velocity[i], swarm[i], personal_best_pos[i], global_best_position)
                    swarm[i] += velocity[i]

                    if np.random.rand() < 0.1:
                        mutation_strength = self.quantum_exponent * (np.abs(bounds[:, 1] - bounds[:, 0]))
                        quantum_mutation = np.random.normal(0, mutation_strength, self.dim)
                        swarm[i] += quantum_mutation * np.random.choice([-1, 1], self.dim)

                    swarm[i] = np.clip(swarm[i], bounds[:, 0], bounds[:, 1])
                    current_value = func(swarm[i])
                    self.eval_count += 1

                    if current_value < personal_best_vals[i]:
                        personal_best_pos[i] = swarm[i]
                        personal_best_vals[i] = current_value

                        if current_value < global_best_value:
                            global_best_position = swarm[i]
                            global_best_value = current_value

                if np.random.rand() < self.temperature:
                    perturbation_strength = np.std(np.abs(swarm - global_best_position), axis=0).mean()
                    perturbation = np.random.normal(0, perturbation_strength, self.dim)
                    candidate = global_best_position + perturbation
                    candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                    candidate_value = func(candidate)
                    self.eval_count += 1

                    if candidate_value < global_best_value:
                        global_best_position = candidate
                        global_best_value = candidate_value

            self.temperature *= self.cooling_rate

        return global_best_position, global_best_value