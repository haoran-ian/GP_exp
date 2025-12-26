import numpy as np

class Enhanced_APSO_GP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)  # Population size scales with budget
        self.inertia = 0.9  # Start with higher inertia
        self.cognitive = 1.5
        self.social = 1.5
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.pop_size, np.inf)
        
        global_best_position = None
        global_best_score = np.inf
        
        for iteration in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                if self.evaluations < self.budget:
                    score = func(particles[i])
                    self.evaluations += 1

                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = particles[i]

                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[i]
                else:
                    break

            # Dynamically adjust inertia
            self.inertia = 0.9 - 0.5 * (iteration / (self.budget // self.pop_size))
            
            # Update velocities and positions
            r1, r2 = np.random.rand(), np.random.rand()
            for i in range(self.pop_size):
                velocities[i] = (self.inertia * velocities[i]
                                 + self.cognitive * r1 * (personal_best_positions[i] - particles[i])
                                 + self.social * r2 * (global_best_position - particles[i]))
                
                particles[i] += velocities[i]
                
                # Adaptive Gaussian perturbation
                perturbation_probability = 0.1 + 0.1 * (iteration / (self.budget // self.pop_size))
                if np.random.rand() < perturbation_probability:
                    particles[i] = np.clip(particles[i] + np.random.normal(0, 0.1, self.dim), lb, ub)
                else:
                    particles[i] = np.clip(particles[i], lb, ub)
        
        return global_best_position, global_best_score