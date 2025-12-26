import numpy as np

class Enhanced_AQ_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(60, budget // 8)  # Increase population size
        self.inertia = 0.6  # Adjusted for better momentum control
        self.cognitive = 2.0  # Enhanced influence of personal best
        self.social_min = 0.5  # Broadened social range for exploration
        self.social_max = 2.5
        self.evaluations = 0
        self.quantum_prob = 0.15  # Increased probability for quantum behavior
        self.delta_scale = 0.9  # Dynamic scaling for velocity step size

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
            
            social = self.social_min + (self.social_max - self.social_min) * np.exp(-iteration / (self.budget / self.pop_size))
            
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (self.inertia * velocities
                          + self.cognitive * r1 * (personal_best_positions - particles)
                          + social * r2 * (global_best_position - particles))
            
            particles += velocities * self.delta_scale  # Scale velocity for dynamic adjustment
            
            if np.random.rand() < self.quantum_prob:  # Apply quantum behavior
                center = (personal_best_positions + global_best_position) / 2
                delta = np.abs(global_best_position - particles)
                particles = center + np.random.uniform(-1, 1, (self.pop_size, self.dim)) * delta / np.sqrt(3)  # Adjusted spread factor
            
            if np.random.rand() < 0.3:  # Increase perturbation probability
                perturbation = np.random.normal(0, 0.15, (self.pop_size, self.dim))  # Enhance perturbation effect
                particles = np.clip(particles + perturbation, lb, ub)
            else:
                particles = np.clip(particles, lb, ub)
        
        return global_best_position, global_best_score