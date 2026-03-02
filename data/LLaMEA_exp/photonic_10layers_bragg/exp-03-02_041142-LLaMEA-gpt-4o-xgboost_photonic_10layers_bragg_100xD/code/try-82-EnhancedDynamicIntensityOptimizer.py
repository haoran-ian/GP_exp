import numpy as np

class EnhancedDynamicIntensityOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.F_min, self.F_max = 0.1, 0.9
        self.CR = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_min, self.w_max = 0.4, 0.9
        self.vel_limit = 0.1
        self.search_space_factor = 0.05
        self.success_rate_threshold = 0.2
        
    def adaptive_operator_selection(self, success_rates):
        normalized_DE = success_rates['DE'] / (success_rates['DE'] + success_rates['PSO'])
        return 'DE' if normalized_DE > self.success_rate_threshold else 'PSO'
    
    def neighborhood_local_search(self, solution, func, lb, ub, neighborhood_size):
        local_best = solution
        local_best_value = func(local_best)
        for _ in range(5):
            candidate = local_best + np.random.uniform(-1, 1, self.dim) * neighborhood_size * (ub - lb)
            candidate = np.clip(candidate, lb, ub)
            candidate_value = func(candidate)
            if candidate_value < local_best_value:
                local_best, local_best_value = candidate, candidate_value
        return local_best, local_best_value
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_pop_size + int(0.1 * self.budget / self.dim)
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        vel = np.zeros((pop_size, self.dim))
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = pop_size
        success_rates = {'DE': 0.5, 'PSO': 0.5}
        
        while evaluations < self.budget:
            selected_operator = self.adaptive_operator_selection(success_rates)
            
            if selected_operator == 'DE':
                successful_updates = 0
                for i in range(pop_size):
                    indices = list(range(i)) + list(range(i + 1, pop_size))
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    F_dynamic = self.F_min + (self.F_max - self.F_min) * np.random.rand()
                    mutant = np.clip(pop[a] + F_dynamic * (pop[b] - pop[c]), lb, ub)
                    trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                    trial_value = func(trial)
                    evaluations += 1
                    
                    if trial_value < personal_best_values[i]:
                        personal_best[i], personal_best_values[i] = trial, trial_value
                        if trial_value < global_best_value:
                            global_best, global_best_value = trial, trial_value
                        successful_updates += 1
                success_rates['DE'] = successful_updates / pop_size
            
            if selected_operator == 'PSO':
                successful_updates = 0
                for i in range(pop_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    w_dynamic = self.w_min + (self.w_max - self.w_min) * np.random.rand()
                    vel[i] = w_dynamic * vel[i] + self.c1 * r1 * (personal_best[i] - pop[i]) + self.c2 * r2 * (global_best - pop[i])
                    vel[i] = np.clip(vel[i], -self.vel_limit * (ub - lb), self.vel_limit * (ub - lb))
                    pop[i] = np.clip(pop[i] + vel[i], lb, ub)
                    
                    new_value = func(pop[i])
                    evaluations += 1
                    if new_value < personal_best_values[i]:
                        personal_best[i], personal_best_values[i] = pop[i], new_value
                        if new_value < global_best_value:
                            global_best, global_best_value = pop[i], new_value
                        successful_updates += 1
                success_rates['PSO'] = successful_updates / pop_size
            
            neighborhood_size = np.exp(-0.01 * evaluations / self.budget)
            global_best, global_best_value = self.neighborhood_local_search(global_best, func, lb, ub, neighborhood_size)
                    
        return global_best