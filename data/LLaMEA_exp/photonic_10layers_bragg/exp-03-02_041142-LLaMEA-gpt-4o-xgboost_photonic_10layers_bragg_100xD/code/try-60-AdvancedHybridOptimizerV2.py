import numpy as np

class AdvancedHybridOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.F = 0.7
        self.CR = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.9
        self.vel_limit = 0.1
        self.search_space_factor = 0.05
        self.operator_success_rates = {'DE': 0.5, 'PSO': 0.5}
        self.operator_rewards = {'DE': 0, 'PSO': 0}
        self.epsilon = 0.1  # Exploration probability for multi-armed bandit

    def select_operator(self):
        # Epsilon-greedy strategy for operator selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(['DE', 'PSO'])
        return max(self.operator_rewards, key=self.operator_rewards.get)

    def update_operator_rewards(self, operator, success_rate):
        # Update rewards using a simple scheme
        self.operator_rewards[operator] = 0.9 * self.operator_rewards[operator] + 0.1 * success_rate

    def local_search(self, solution, func, lb, ub):
        local_best = solution
        local_best_value = func(local_best)
        for _ in range(5):
            candidate = local_best + np.random.uniform(-1, 1, self.dim) * self.search_space_factor * (ub - lb)
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

        while evaluations < self.budget:
            selected_operator = self.select_operator()
            successful_updates = 0

            if selected_operator == 'DE':
                for i in range(pop_size):
                    indices = list(range(i)) + list(range(i + 1, pop_size))
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), lb, ub)
                    trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                    trial_value = func(trial)
                    evaluations += 1

                    if trial_value < personal_best_values[i]:
                        personal_best[i], personal_best_values[i] = trial, trial_value
                        if trial_value < global_best_value:
                            global_best, global_best_value = trial, trial_value
                        successful_updates += 1
                self.F = 0.5 + 0.5 * np.random.rand()

            if selected_operator == 'PSO':
                for i in range(pop_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    vel[i] = self.w * vel[i] + self.c1 * r1 * (personal_best[i] - pop[i]) + self.c2 * r2 * (global_best - pop[i])
                    vel[i] = np.clip(vel[i], -self.vel_limit * (ub - lb), self.vel_limit * (ub - lb))
                    pop[i] = np.clip(pop[i] + vel[i], lb, ub)

                    new_value = func(pop[i])
                    evaluations += 1
                    if new_value < personal_best_values[i]:
                        personal_best[i], personal_best_values[i] = pop[i], new_value
                        if new_value < global_best_value:
                            global_best, global_best_value = pop[i], new_value
                        successful_updates += 1
                self.w = 0.4 + 0.5 * np.random.rand()

            success_rate = successful_updates / pop_size
            self.update_operator_rewards(selected_operator, success_rate)

            # Apply local search to the best solution found so far
            global_best, global_best_value = self.local_search(global_best, func, lb, ub)

        return global_best