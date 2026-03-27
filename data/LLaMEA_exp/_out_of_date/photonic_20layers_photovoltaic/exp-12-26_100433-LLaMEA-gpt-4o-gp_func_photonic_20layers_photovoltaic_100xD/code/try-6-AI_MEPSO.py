import numpy as np

class AI_MEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.social_min = 1.0
        self.social_max = 3.0
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.pop_size, np.inf)
        
        global_best_position = None
        global_best_score = np.inf
        
        iteration = 0
        while self.evaluations < self.budget:
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
            
            w = self.w_max - ((self.w_max - self.w_min) * (iteration / (self.budget // self.pop_size)))
            social = self.social_min + (self.social_max - self.social_min) * np.exp(-iteration / (self.budget / self.pop_size))
            
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (w * velocities
                          + self.c1 * r1 * (personal_best_positions - particles)
                          + social * r2 * (global_best_position - particles))

            particles += velocities
            
            if self.evaluations / self.budget < 0.5 and np.random.rand() < 0.3:  # 30% chance for mutation in early stages
                mutation = np.random.normal(0, 0.05, (self.pop_size, self.dim))
                particles = np.clip(particles + mutation, lb, ub)
            else:
                particles = np.clip(particles, lb, ub)

            iteration += 1
        
        return global_best_position, global_best_score