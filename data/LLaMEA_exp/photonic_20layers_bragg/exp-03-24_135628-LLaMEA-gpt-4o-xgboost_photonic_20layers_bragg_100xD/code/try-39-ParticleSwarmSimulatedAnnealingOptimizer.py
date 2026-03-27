import numpy as np

class ParticleSwarmSimulatedAnnealingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        lb, ub = func.bounds.lb, func.bounds.ub
        num_particles = min(50, self.budget // 2)
        particles = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_fitness = np.array([func(p) for p in particles])
        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index]
        global_best_fitness = personal_best_fitness[global_best_index]

        evaluations = num_particles
        inertia_weight = 0.7
        cognitive_coeff = 1.5
        social_coeff = 1.5

        while evaluations < self.budget:
            T = max(0.01, 1.0 - evaluations / self.budget)
            for i in range(num_particles):
                # PSO velocity and position update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = cognitive_coeff * r1 * (personal_best_positions[i] - particles[i])
                social_velocity = social_coeff * r2 * (global_best_position - particles[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_velocity + social_velocity
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                # Fitness evaluation
                current_fitness = func(particles[i])
                evaluations += 1

                # Update personal bests
                if current_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = current_fitness
                    personal_best_positions[i] = particles[i]

                # Update global best
                if current_fitness < global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = particles[i]

            # Adaptive Simulated Annealing-like acceptance
            for i in range(num_particles):
                new_candidate = personal_best_positions[i] + np.random.normal(0, 0.1, self.dim)
                new_candidate = np.clip(new_candidate, lb, ub)
                new_fitness = func(new_candidate)
                evaluations += 1
                if new_fitness < personal_best_fitness[i] or np.random.rand() < np.exp((personal_best_fitness[i] - new_fitness) / T):
                    personal_best_positions[i] = new_candidate
                    personal_best_fitness[i] = new_fitness

                    if new_fitness < global_best_fitness:
                        global_best_fitness = new_fitness
                        global_best_position = new_candidate

            if evaluations >= self.budget:
                break

        return global_best_position