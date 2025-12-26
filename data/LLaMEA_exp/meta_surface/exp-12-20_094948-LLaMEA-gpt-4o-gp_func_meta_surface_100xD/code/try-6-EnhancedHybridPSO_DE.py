import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.F_max = 1.0
        self.F_min = 0.5
        self.CR = 0.9
        self.epsilon = 1e-8
        self.neighborhood_size = 5  # Dynamic neighborhood size

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        
        eval_count = self.pop_size

        while eval_count < self.budget:
            diversity = np.mean(np.std(swarm, axis=0))
            self.w = self.w_max - (self.w_max - self.w_min) * (diversity / (diversity + self.epsilon))
            
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                                 self.c2 * r2 * (global_best_position - swarm[i]))
                swarm[i] = np.clip(swarm[i] + velocities[i], lower_bound, upper_bound)
            
            for i in range(self.pop_size):
                neighbors = np.random.choice(self.pop_size, self.neighborhood_size, replace=False)
                F_adaptive = self.F_max - (self.F_max - self.F_min) * (diversity / (diversity + self.epsilon))
                a, b, c = np.random.choice(neighbors, 3, replace=False)
                mutant = np.clip(swarm[a] + F_adaptive * (swarm[b] - swarm[c]), lower_bound, upper_bound)
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, swarm[i])
                trial_score = func(trial)
                eval_count += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score
                    if trial_score < func(global_best_position):
                        global_best_position = trial
            if eval_count >= self.budget:
                break
        
        return global_best_position