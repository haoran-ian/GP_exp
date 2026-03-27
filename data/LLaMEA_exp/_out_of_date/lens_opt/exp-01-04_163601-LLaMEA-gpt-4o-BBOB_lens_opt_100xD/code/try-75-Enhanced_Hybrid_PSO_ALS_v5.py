import numpy as np

class Enhanced_Hybrid_PSO_ALS_v5:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 40  # Increased for better exploration
        self.c1 = 2.0  # Adjusted cognitive component
        self.c2 = 2.0  # Adjusted social component
        self.w_max = 0.9
        self.w_min = 0.4
        self.mutation_rate = 0.2  # Increased mutation rate
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.subpop_size_factor = 0.2  # Larger subpopulations for diversity
        self.competition_factor = 0.3  # Probability of competition amongst subpopulations

    class Particle:
        def __init__(self, dim, bounds):
            self.position = np.random.uniform(bounds.lb, bounds.ub, dim)
            self.velocity = np.zeros(dim)
            self.best_position = np.copy(self.position)
            self.best_value = float('inf')

    def __call__(self, func):
        bounds = func.bounds
        self.particles = [self.Particle(self.dim, bounds) for _ in range(self.num_particles)]

        evaluations = 0
        while evaluations < self.budget:
            # Calculate adaptive inertia weight
            self.w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            
            # Adaptive subpopulation size and competition
            self.subpop_size = max(2, int(self.num_particles * self.subpop_size_factor))
            subpop_indices = np.random.permutation(self.num_particles).reshape(-1, self.subpop_size)
            for indices in subpop_indices:
                subpop_best_value = float('inf')
                subpop_best_position = None
                for i in indices:
                    particle = self.particles[i]
                    fitness_value = func(particle.position)
                    evaluations += 1

                    if fitness_value < particle.best_value:
                        particle.best_value = fitness_value
                        particle.best_position = np.copy(particle.position)

                    if fitness_value < self.global_best_value:
                        self.global_best_value = fitness_value
                        self.global_best_position = np.copy(particle.position)

                    if fitness_value < subpop_best_value:
                        subpop_best_value = fitness_value
                        subpop_best_position = np.copy(particle.position)

                    if evaluations >= self.budget:
                        break

                    # Update particle velocity and position
                    r1, r2 = np.random.uniform(size=2)
                    cognitive_velocity = self.c1 * r1 * (particle.best_position - particle.position)
                    social_velocity = self.c2 * r2 * (self.global_best_position - particle.position)
                    particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                    particle.position = particle.position + particle.velocity

                    # Constrain to bounds
                    particle.position = np.clip(particle.position, bounds.lb, bounds.ub)

                    # Introduce diversity preservation via mutation
                    if np.random.rand() < self.mutation_rate * (1 - evaluations / self.budget):
                        particle.position = np.clip(particle.position + np.random.normal(0, 0.1, self.dim), bounds.lb, bounds.ub)

                # Subpopulation competition: promote exploration
                if np.random.rand() < self.competition_factor:
                    for i in indices:
                        if np.random.rand() < 0.5:  # Random chance to adopt the subpopulation best
                            self.particles[i].position = np.copy(subpop_best_position)

        return self.global_best_position, self.global_best_value