import numpy as np
from scipy.optimize import minimize

class AdaptiveSuccessRateTuning:
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
        self.success_threshold = 0.1
        self.success_rate = 0.0
        self.recent_successes = []
        self.feedback_window = 5
    
    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.power((np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)) /
                (np.math.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta))
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.power(np.abs(v), 1 / beta)
        return step
    
    def local_search(self, func, point):
        result = minimize(func, point, method='Powell')
        return result.x, result.fun

    def restructure_subpopulations(self, populations, scores):
        combined_pop = np.concatenate(populations)
        combined_scores = np.concatenate(scores)
        sorted_indices = np.argsort(combined_scores)
        top_individuals = combined_pop[sorted_indices][:self.pop_size]
        return np.array_split(top_individuals, self.num_subpops)

    def update_success_rate(self, improvement):
        self.recent_successes.append(improvement)
        if len(self.recent_successes) > self.feedback_window:
            self.recent_successes.pop(0)
        self.success_rate = np.mean(self.recent_successes) if self.recent_successes else 0

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

                self.w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
                self.F = self.F_min + self.success_rate * (self.F_max - self.F_min)
                self.CR = self.CR_base + 0.2 * self.success_rate
                
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
                    improvement = max(0, pbest_score[i] - trial_score)
                    self.update_success_rate(improvement)
                    if trial_score < pbest_score[i]:
                        pbest_pos[i] = trial
                        pbest_score[i] = trial_score
                        if trial_score < global_best_score:
                            global_best_position = trial
                            global_best_score = trial_score

                if np.random.rand() < 0.3 * (1 - self.success_rate):  # Adjusted local search frequency
                    for i in range(self.subpop_size):
                        refined_position, refined_score = self.local_search(func, swarm[i])
                        eval_count += refined_position.size
                        if refined_score < pbest_score[i]:
                            pbest_pos[i] = refined_position
                            pbest_score[i] = refined_score
                            if refined_score < global_best_score:
                                global_best_position = refined_position
                                global_best_score = refined_score

            populations = self.restructure_subpopulations(populations, personal_best_scores)

            if eval_count >= self.budget:
                break
        
        return global_best_position