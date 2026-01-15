import numpy as np

class MultiSwarmAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        # Initialize parameters
        num_particles = 15
        num_swarms = 3
        particles_per_swarm = num_particles // num_swarms
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]
        max_velocity = 0.5
        cognitive_weight = 2.0
        social_weight = 1.8
        inertia_weight = 0.729
        regroup_freq = 20

        while self.evaluations < self.budget:
            # Regroup particles into new swarms to enhance exploration
            if self.evaluations % regroup_freq == 0:
                np.random.shuffle(particles)

            for swm in range(num_swarms):
                swarm_start = swm * particles_per_swarm
                swarm_end = swarm_start + particles_per_swarm
                swarm_particles = particles[swarm_start:swarm_end]
                swarm_best_index = swarm_start + np.argmin(personal_best_values[swarm_start:swarm_end])
                swarm_best_pos = personal_best_positions[swarm_best_index]

                for idx in range(swarm_start, swarm_end):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    inertia = inertia_weight * velocities[idx]
                    cognitive = cognitive_weight * r1 * (personal_best_positions[idx] - particles[idx])
                    social = social_weight * r2 * (swarm_best_pos - particles[idx])

                    velocities[idx] = inertia + cognitive + social
                    velocities[idx] = np.clip(velocities[idx], -max_velocity, max_velocity)
                    particles[idx] += velocities[idx]
                    particles[idx] = np.clip(particles[idx], func.bounds.lb, func.bounds.ub)

                    value = func(particles[idx])
                    self.evaluations += 1

                    if value < personal_best_values[idx]:
                        personal_best_values[idx] = value
                        personal_best_positions[idx] = particles[idx]
                    if value < global_best_value:
                        global_best_value = value
                        global_best_position = particles[idx]

                    if self.evaluations >= self.budget:
                        break
                
                if self.evaluations >= self.budget:
                    break

        return global_best_position, global_best_value