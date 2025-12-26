import numpy as np
from scipy.optimize import minimize

class AdaptiveHistoricalMultiPop:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.subpop_size = 5
        self.num_subpops = self.pop_size // self.subpop_size
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_max = 2.5
        self.c1_min = 1.0
        self.c2_max = 2.5
        self.c2_min = 1.0
        self.F_max = 1.5
        self.F_min = 0.5
        self.CR_base = 0.7
        self.epsilon = 1e-8
        self.improvement_threshold = 0.01
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
        result = minimize(func, point, method='Powell')
        return result.x, result.fun

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        populations = [np.random.uniform(lower_bound, upper_bound, (self.subpop_size, self.dim)) for _ in range(self.num_subpops)]
        velocities = [np.random.uniform(-1, 1, (self.subpop_size, self.dim)) for _ in range(self.num_subpops)]
        personal_best_positions = [np.copy(pop) for pop in populations]
        personal_best_scores = [np.array([func(ind) for ind in pop]) for pop in populations]
        global_best_position = personal_best_positions[0][np.argmin(personal_best_scores[0])]
        global_best_score = func(global_best_position)
        
        eval_count = sum(len(scores) for scores in personal_best_scores)

        while eval_count < self.budget:
            for pop_idx in range(self.num_subpops):
                swarm = populations[pop_idx]
                velocity = velocities[pop_idx]
                pbest_pos = personal_best_positions[pop_idx]
                pbest_score = personal_best_scores[pop_idx]

                diversity = np.mean(np.std(swarm, axis=0))
                self.w = self.w_max - (self.w_max - self.w_min) * (diversity / (diversity + self.epsilon))
                self.chaos_factor = (4 * self.chaos_factor) * (1 - self.chaos_factor)
                self.F = self.F_min + self.chaos_factor * (self.F_max - self.F_min)
                self.CR = self.CR_base + 0.2 * self.chaos_factor
                
                self.c1 = self.c1_max - (self.c1_max - self.c1_min) * (eval_count / self.budget)
                self.c2 = self.c2_min + (self.c2_max - self.c2_min) * (eval_count / self.budget)

                r1, r2 = np.random.rand(self.subpop_size, self.dim), np.random.rand(self.subpop_size, self.dim)
                velocity = (self.w * velocity +
                            self.c1 * r1 * (pbest_pos - swarm) +
                            self.c2 * r2 * (global_best_position - swarm))
                swarm = np.clip(swarm + velocity, lower_bound, upper_bound)
                
                if np.random.rand() < 0.5:
                    swarm += self.levy_flight((self.subpop_size, self.dim))
                
                for i in range(self.subpop_size):
                    indices = [idx for idx in range(self.subpop_size) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = np.clip(swarm[a] + self.F * (swarm[b] - swarm[c]), lower_bound, upper_bound)
                    crossover = np.random.rand(self.dim) < self.CR
                    if not np.any(crossover):
                        crossover[np.random.randint(0, self.dim)] = True
                    trial = np.where(crossover, mutant, swarm[i])
                    trial_score = func(trial)
                    eval_count += 1
                    if trial_score < pbest_score[i]:
                        pbest_pos[i] = trial
                        pbest_score[i] = trial_score
                        if trial_score < global_best_score:
                            self.recent_improvements.append(global_best_score - trial_score)
                            global_best_position = trial + np.random.normal(0, 0.1, self.dim)  # Perturbation added here
                            global_best_score = trial_score
                            if len(self.recent_improvements) > self.feedback_window:
                                self.recent_improvements.pop(0)

                if np.random.rand() < 0.2 * (1 - diversity):  # Increased local search frequency
                    for i in range(self.subpop_size):
                        refined_position, refined_score = self.local_search(func, swarm[i])
                        eval_count += refined_position.size
                        if refined_score < pbest_score[i]:
                            pbest_pos[i] = refined_position
                            pbest_score[i] = refined_score
                            if refined_score < global_best_score:
                                global_best_position = refined_position
                                global_best_score = refined_score

            if len(self.recent_improvements) >= self.feedback_window and np.mean(self.recent_improvements) < self.improvement_threshold:
                self.c1 = np.clip(self.c1 + 0.1, 0, 2.5)
                self.c2 = np.clip(self.c2 - 0.1, 0, 2.5)
                self.recent_improvements = []

            if eval_count >= self.budget:
                break
        
        return global_best_position