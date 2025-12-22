import numpy as np

class EnhancedDAPSOPlus:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget // 10)
        self.c1_start, self.c1_end = 2.5, 0.5   # dynamic cognitive component
        self.c2_start, self.c2_end = 0.5, 2.5   # dynamic social component
        self.w_start, self.w_end = 0.9, 0.4     # dynamic inertia weight
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.adaptive_learning_rate = 0.1  # initial adaptive learning rate
    
    def chaotically_mutate(self, position, lb, ub, progress):
        beta = 0.6  # mutation intensity
        chaos_factor = 0.5 + 3.5 * progress * (1 - progress)
        mutation = beta * chaos_factor * (np.random.rand(self.dim) - 0.5)
        return np.clip(position + mutation, lb, ub)
    
    def promote_diversity(self, positions, lb, ub):
        max_dist = np.linalg.norm(ub - lb)
        for i in range(len(positions)):
            if np.random.rand() < 0.1:
                random_direction = np.random.uniform(-1, 1, self.dim)
                random_distance = np.random.uniform(0, max_dist) * self.adaptive_learning_rate
                positions[i] = np.clip(positions[i] + random_direction * random_distance, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particle_positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        particle_velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        personal_best_positions = particle_positions.copy()
        personal_best_values = np.full(self.population_size, float('inf'))

        evaluations = 0

        while evaluations < self.budget:
            subgroup_size = max(2, int((np.sin((evaluations / self.budget) * np.pi) + 1) / 2 * self.population_size))
            subgroup_indices = np.random.choice(self.population_size, subgroup_size, replace=False)
            subgroup_best_position, subgroup_best_value = None, float('inf')

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                current_value = func(particle_positions[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particle_positions[i]

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = particle_positions[i]

                if i in subgroup_indices and current_value < subgroup_best_value:
                    subgroup_best_value = current_value
                    subgroup_best_position = particle_positions[i]

            progress = evaluations / self.budget
            c1 = self.c1_start * (1 - progress) + self.c1_end * progress
            c2 = self.c2_start * (1 - progress) + self.c2_end * progress
            w = self.w_start * (1 - progress) + self.w_end * progress

            self.adaptive_learning_rate = max(0.05, (self.global_best_value - np.mean(personal_best_values)) / (np.std(personal_best_values) + 1e-9))

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = c1 * r1 * (personal_best_positions[i] - particle_positions[i])
                social_velocity = c2 * r2 * (subgroup_best_position - particle_positions[i] if i in subgroup_indices else self.global_best_position - particle_positions[i])
                particle_velocities[i] = w * particle_velocities[i] + cognitive_velocity + social_velocity
                particle_positions[i] += particle_velocities[i]

                particle_positions[i] = self.chaotically_mutate(particle_positions[i], lb, ub, progress)

            self.promote_diversity(particle_positions, lb, ub)

        return self.global_best_position, self.global_best_value