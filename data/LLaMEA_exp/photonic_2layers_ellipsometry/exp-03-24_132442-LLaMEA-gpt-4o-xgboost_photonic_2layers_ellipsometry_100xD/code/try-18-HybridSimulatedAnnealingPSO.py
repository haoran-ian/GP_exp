import numpy as np

class HybridSimulatedAnnealingPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 5

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters
        T_initial = 1.0
        T_final = 0.001
        alpha = 0.9
        
        # Particle Swarm Optimization (PSO) Parameters
        inertia_weight = 0.7
        cognitive_weight = 1.5
        social_weight = 1.5

        # Initialize particles
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        particle_velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        particle_best_positions = np.copy(particles)
        particle_best_values = np.array([func(p) for p in particles])
        
        # Global best
        global_best_idx = np.argmin(particle_best_values)
        global_best_position = np.copy(particles[global_best_idx])
        global_best_value = particle_best_values[global_best_idx]
        
        evaluations = self.population_size

        # Define dual annealing schedules
        schedule_A = lambda evals: T_initial * (T_final / T_initial) ** (evals / self.budget)
        schedule_B = lambda evals: T_initial * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))

        # Hybrid Simulated Annealing PSO Loop
        while evaluations < self.budget:
            # Toggle between two annealing schedules
            if evaluations % 2 == 0:
                T = schedule_A(evaluations)
            else:
                T = schedule_B(evaluations)
            
            for i in range(self.population_size):
                # Simulated Annealing Perturbation
                scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** (1 / alpha))
                perturbation_factor = 1 + 0.1 * np.sin(np.pi * evaluations / self.budget)
                perturbation = np.random.normal(0, scale / (5 * perturbation_factor), self.dim)
                candidate_solution = particles[i] + perturbation
                candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
                candidate_value = func(candidate_solution)
                evaluations += 1

                # Metropolis criterion
                if candidate_value < particle_best_values[i] or np.random.rand() < np.exp((particle_best_values[i] - candidate_value) / T):
                    particles[i] = candidate_solution
                    particle_best_positions[i] = candidate_solution
                    particle_best_values[i] = candidate_value

                    # Update global best
                    if candidate_value < global_best_value:
                        global_best_position = candidate_solution
                        global_best_value = candidate_value

                # PSO Update Velocities
                r1, r2 = np.random.rand(2, self.dim)
                particle_velocities[i] = (inertia_weight * particle_velocities[i] +
                                          cognitive_weight * r1 * (particle_best_positions[i] - particles[i]) +
                                          social_weight * r2 * (global_best_position - particles[i]))
                particles[i] = np.clip(particles[i] + particle_velocities[i], func.bounds.lb, func.bounds.ub)

        return global_best_position, global_best_value