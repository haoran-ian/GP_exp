import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive parameter - modified for better exploitation
        self.c2 = 2.0  # social parameter
        self.F = 0.5  # DE scaling factor
        self.CR = 0.9  # DE crossover probability

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        
        # Initialize the positions and velocities of particles
        particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 5
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        evaluations = self.population_size

        while evaluations < self.budget:
            # Update velocities and positions using PSO and DE strategies
            for i in range(self.population_size):
                # PSO update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                
                # DE Mutation and Crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = particles[indices]
                mutant_vector = x1 + self.F * (x2 - x3)
                crossover = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover, mutant_vector, particles[i])
                
                # Boundary handling
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                trial_vector = np.clip(trial_vector, lb, ub)

                # Evaluate the trial vector
                trial_score = func(trial_vector)
                evaluations += 1
                
                # Update personal best
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score
                    
                    # Update global best
                    if trial_score < global_best_score:
                        global_best_position = trial_vector
                        global_best_score = trial_score

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_score