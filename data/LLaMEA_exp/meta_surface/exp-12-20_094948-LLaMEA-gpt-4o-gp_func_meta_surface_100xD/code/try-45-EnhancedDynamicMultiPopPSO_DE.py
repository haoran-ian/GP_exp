import numpy as np
from scipy.optimize import minimize

class EnhancedDynamicMultiPopPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.num_subpops = 2
        self.subpop_size = self.pop_size // self.num_subpops
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_max = 2.0
        self.c1_min = 1.2
        self.c2_max = 2.0
        self.c2_min = 1.2
        self.F_max = 1.0
        self.F_min = 0.5
        self.CR_base = 0.9
        self.epsilon = 1e-8
        self.improvement_threshold = 0.005
        self.recent_improvements = []
        self.feedback_window = 5
        self.chaos_factor = np.random.rand()
        self.levy_beta = 1.5

    def levy_flight(self, size):
        sigma = (np.power((np.math.gamma(1 + self.levy_beta) * np.sin(np.pi * self.levy_beta / 2)) /
                (np.math.gamma((1 + self.levy_beta) / 2) * self.levy_beta * np.power(2, (self.levy_beta - 1) / 2)), 1 / self.levy_beta))
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.power(np.abs(v), 1 / self.levy_beta)
        return step
    
    def local_search(self, func, point):
        result = minimize(func, point, method='Nelder-Mead')
        return result.x, result.fun

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = func(global_best_position)
        
        eval_count = self.pop_size

        while eval_count < self.budget:
            diversity = np.mean(np.std(swarm, axis=0))
            self.w = self.w_max - (self.w_max - self.w_min) * (diversity / (diversity + self.epsilon))
            self.chaos_factor = (4 * self.chaos_factor) * (1 - self.chaos_factor)
            self.F = self.F_min + self.chaos_factor * (self.F_max - self.F_min)
            self.CR = self.CR_base + 0.1 * self.chaos_factor
            
            # Dynamic parameter adaptation
            self.c1 = self.c1_max - (self.c1_max - self.c1_min) * (eval_count / self.budget)
            self.c2 = self.c2_min + (self.c2_max - self.c2_min) * (eval_count / self.budget)
            
            # PSO Update with multi-population approach
            for subpop_idx in range(self.num_subpops):
                start_idx = subpop_idx * self.subpop_size
                end_idx = start_idx + self.subpop_size
                subpop = swarm[start_idx:end_idx]
                subpop_velocities = velocities[start_idx:end_idx]
                personal_best_subpop = personal_best_positions[start_idx:end_idx]
                personal_best_subpop_scores = personal_best_scores[start_idx:end_idx]
                
                r1, r2 = np.random.rand(self.subpop_size, self.dim), np.random.rand(self.subpop_size, self.dim)
                subpop_velocities = (self.w * subpop_velocities +
                                     self.c1 * r1 * (personal_best_subpop - subpop) +
                                     self.c2 * r2 * (global_best_position - subpop))
                subpop = np.clip(subpop + subpop_velocities, lower_bound, upper_bound)
                
                # Enhanced Lévy flight for improved exploration
                if np.random.rand() < 0.5:
                    subpop += self.levy_flight((self.subpop_size, self.dim))
                
                # DE Update
                for i in range(self.subpop_size):
                    indices = [idx for idx in range(self.subpop_size) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = np.clip(subpop[a] + self.F * (subpop[b] - subpop[c]), lower_bound, upper_bound)
                    crossover = np.random.rand(self.dim) < self.CR + 0.1  # Modified crossover
                    if not np.any(crossover):
                        crossover[np.random.randint(0, self.dim)] = True
                    trial = np.where(crossover, mutant, subpop[i])
                    trial_score = func(trial)
                    eval_count += 1
                    if trial_score < personal_best_subpop_scores[i]:
                        personal_best_subpop[i] = trial
                        personal_best_subpop_scores[i] = trial_score
                        if trial_score < global_best_score:
                            self.recent_improvements.append(global_best_score - trial_score)
                            global_best_position = trial
                            global_best_score = trial_score
                            if len(self.recent_improvements) > self.feedback_window:
                                self.recent_improvements.pop(0)

                swarm[start_idx:end_idx] = subpop
                velocities[start_idx:end_idx] = subpop_velocities
                personal_best_positions[start_idx:end_idx] = personal_best_subpop
                personal_best_scores[start_idx:end_idx] = personal_best_subpop_scores
                
            # Adaptive local search phase
            if np.random.rand() < 0.1 * (1 - diversity):
                for i in range(self.pop_size):
                    refined_position, refined_score = self.local_search(func, swarm[i])
                    eval_count += refined_position.size
                    if refined_score < personal_best_scores[i]:
                        personal_best_positions[i] = refined_position
                        personal_best_scores[i] = refined_score
                        if refined_score < global_best_score:
                            global_best_position = refined_position
                            global_best_score = refined_score
            
            if len(self.recent_improvements) >= self.feedback_window and np.mean(self.recent_improvements) < self.improvement_threshold:
                self.c1 = np.clip(self.c1 + 0.1, 0, 2.0)
                self.c2 = np.clip(self.c2 - 0.1, 0, 2.0)
                self.recent_improvements = []

            # Regroup populations if necessary
            if eval_count % (self.pop_size * 5) == 0:
                swarm = np.random.permutation(swarm)
                velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))

            if eval_count >= self.budget:
                break
        
        return global_best_position