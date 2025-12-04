import numpy as np

class RefinedHybridPSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 50
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 1.5
        self.temperature = 1000.0
        self.cooling_rate = 0.9
        self.dynamic_particles = True
    
    def __call__(self, func):
        np.random.seed(42)
        particles_position = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        particles_velocity = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_position = np.copy(particles_position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.num_particles
        
        while evaluations < self.budget:
            self.inertia_weight *= (0.99 - 0.01 * (evaluations/self.budget))
            adaptive_cognitive_coefficient = self.cognitive_coefficient * (evaluations / self.budget)
            adaptive_social_coefficient = self.social_coefficient * (1 - evaluations / self.budget)
            velocity_scaling_factor = 1 - (evaluations / self.budget)
            
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                particles_velocity[i] = (
                    self.inertia_weight * particles_velocity[i] +
                    adaptive_cognitive_coefficient * r1 * (personal_best_position[i] - particles_position[i]) +
                    adaptive_social_coefficient * r2 * (global_best_position - particles_position[i])
                ) * velocity_scaling_factor
                particles_position[i] += particles_velocity[i]
                particles_position[i] = np.clip(particles_position[i], self.lower_bound, self.upper_bound)
                
                new_value = func(particles_position[i])
                evaluations += 1

                if new_value < personal_best_value[i]:
                    personal_best_position[i] = particles_position[i]
                    personal_best_value[i] = new_value

                if new_value < global_best_value:
                    global_best_position = particles_position[i]
                    global_best_value = new_value

            for i in range(self.num_particles):
                candidate_position = particles_position[i] + np.random.normal(0, 1, self.dim)
                candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                candidate_value = func(candidate_position)
                evaluations += 1

                if candidate_value < personal_best_value[i] or np.exp((personal_best_value[i] - candidate_value) / self.temperature) > np.random.rand():
                    personal_best_position[i] = candidate_position
                    personal_best_value[i] = candidate_value

                if candidate_value < global_best_value:
                    global_best_position = candidate_position
                    global_best_value = candidate_value

            self.temperature *= self.cooling_rate

            if evaluations % (self.budget // 10) == 0 and self.dynamic_particles:
                self.num_particles = max(10, self.num_particles - 1)

            if evaluations % (self.budget // 5) == 0:
                indices = np.random.choice(self.num_particles, size=self.num_particles//3, replace=False)
                for idx in indices:
                    particles_position[idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

            if evaluations >= self.budget:
                break

        return global_best_position, global_best_value