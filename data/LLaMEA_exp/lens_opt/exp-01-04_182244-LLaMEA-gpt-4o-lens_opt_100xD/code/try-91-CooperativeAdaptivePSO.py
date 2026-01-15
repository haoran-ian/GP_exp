import numpy as np

class CooperativeAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
    
    def __call__(self, func):
        # Initialize parameters
        num_particles = 15
        num_swarms = 3
        particles = [np.random.uniform(func.bounds.lb, func.bounds.ub, (num_particles, self.dim)) for _ in range(num_swarms)]
        velocities = [np.random.uniform(-1, 1, (num_particles, self.dim)) for _ in range(num_swarms)]
        personal_best_positions = [p.copy() for p in particles]
        personal_best_values = [np.array([func(p) for p in swarm]) for swarm in particles]
        
        global_best_position = min(personal_best_positions, key=lambda swarm: np.min([func(x) for x in swarm]))
        global_best_value = min([np.min(vals) for vals in personal_best_values])
        
        initial_radius = 0.1
        max_velocity = 0.6
        adaptive_mutation_freq = 8
        inertia_weight = 0.9
        
        def _dynamic_weight(evals):
            return 0.5 + 0.5 * np.cos(np.pi * evals / self.budget)
        
        while self.evaluations < self.budget:
            for s in range(num_swarms):
                for idx in range(num_particles):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    inertia = inertia_weight * _dynamic_weight(self.evaluations)
                    
                    cognitive = 2.4 * r1 * (personal_best_positions[s][idx] - particles[s][idx])
                    social = 1.8 * r2 * (global_best_position - particles[s][idx])
                    
                    velocities[s][idx] = inertia * velocities[s][idx] + cognitive + 0.5 * social
                    velocities[s][idx] = np.clip(velocities[s][idx], -max_velocity, max_velocity)
                    particles[s][idx] += velocities[s][idx]
                    particles[s][idx] = np.clip(particles[s][idx], func.bounds.lb, func.bounds.ub)

                    if idx == np.argmin(personal_best_values[s]):
                        angle = self.evaluations * 0.15 + np.random.normal(0, 0.05)
                        displacement = np.array([
                            initial_radius * np.cos(angle),
                            initial_radius * np.sin(angle)
                        ] + [0] * (self.dim - 2))
                        spiral_position = global_best_position + displacement[:self.dim]
                        spiral_position = np.clip(spiral_position, func.bounds.lb, func.bounds.ub)
                        particles[s][idx] = spiral_position

                    value = func(particles[s][idx])
                    self.evaluations += 1

                    if value < personal_best_values[s][idx]:
                        personal_best_values[s][idx] = value
                        personal_best_positions[s][idx] = particles[s][idx]
                    if value < global_best_value:
                        global_best_value = value
                        global_best_position = particles[s][idx]

                    if self.evaluations % adaptive_mutation_freq == 0:
                        particles[s][idx] += np.random.normal(0, 0.03 + 0.1 * (self.evaluations / self.budget), self.dim)
                        particles[s][idx] = np.clip(particles[s][idx], func.bounds.lb, func.bounds.ub)

                    if self.evaluations >= self.budget:
                        break
            
        return global_best_position, global_best_value