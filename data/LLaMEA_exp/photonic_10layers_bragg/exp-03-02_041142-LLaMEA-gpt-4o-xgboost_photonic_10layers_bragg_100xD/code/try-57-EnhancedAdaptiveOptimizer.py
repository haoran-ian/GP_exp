import numpy as np

class EnhancedAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.success_rate_threshold = 0.2
        self.local_search_iterations = 10  # Increased for deeper exploitation
        self.phase_switch_threshold = 0.6  # Threshold for switching exploration/exploitation phase
    
    def adaptive_operator_selection(self, success_rates):
        if success_rates['DE'] > success_rates['PSO']:
            return 'DE'
        return 'PSO'
    
    def local_search(self, solution, func, lb, ub):
        local_best = solution
        local_best_value = func(local_best)
        for _ in range(self.local_search_iterations):
            candidate = local_best + np.random.uniform(-0.5, 0.5, self.dim) * (ub - lb) * 0.01  # Refined local search step
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
        exploration_phase = True
        
        while evaluations < self.budget:
            selected_operator = self.adaptive_operator_selection(success_rates)
            
            if selected_operator == 'DE':
                successful_updates = 0
                for i in range(pop_size):
                    indices = list(range(i)) + list(range(i + 1, pop_size))
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    if exploration_phase:
                        F = 0.9  # Stronger perturbation for exploration
                    else:
                        F = 0.5  # Finer perturbation for exploitation
                    mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), lb, ub)
                    trial = np.where(np.random.rand(self.dim) < 0.9, mutant, pop[i])
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
                    if exploration_phase:
                        inertia_weight = 0.9  # Higher inertia for broader search
                    else:
                        inertia_weight = 0.4  # Lower inertia for focused search
                    vel[i] = inertia_weight * vel[i] + 2.0 * r1 * (personal_best[i] - pop[i]) + 2.0 * r2 * (global_best - pop[i])
                    vel[i] = np.clip(vel[i], -0.1 * (ub - lb), 0.1 * (ub - lb))
                    pop[i] = np.clip(pop[i] + vel[i], lb, ub)
                    
                    new_value = func(pop[i])
                    evaluations += 1
                    if new_value < personal_best_values[i]:
                        personal_best[i], personal_best_values[i] = pop[i], new_value
                        if new_value < global_best_value:
                            global_best, global_best_value = pop[i], new_value
                        successful_updates += 1
                success_rates['PSO'] = successful_updates / pop_size
            
            if (evaluations / self.budget) > self.phase_switch_threshold:
                exploration_phase = False
            
            global_best, global_best_value = self.local_search(global_best, func, lb, ub)
                    
        return global_best