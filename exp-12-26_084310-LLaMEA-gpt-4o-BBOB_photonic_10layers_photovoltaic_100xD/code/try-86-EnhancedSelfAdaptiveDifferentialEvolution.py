import numpy as np

class EnhancedSelfAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9  # Initial crossover rate
        self.strategy_prob = [0.5, 0.5]
        self.success_rate = [0, 0]
        self.recent_success = [0, 0]  # Change: Track recent success
        self.learning_rate = 0.1
        self.pop_shrink_factor = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_pop_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx]
        best_fitness = fitness[best_idx]
        eval_count = pop_size
        
        while eval_count < self.budget:
            new_pop = []
            new_fitness = []
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                
                strategy = np.random.choice([0, 1], p=self.strategy_prob)
                
                idxs = [idx for idx in range(pop_size) if idx != i]
                randomized_F = np.random.uniform(0.1, 1.0)
                if strategy == 0:
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + randomized_F * (b - c) * (1 + self.success_rate[strategy] - self.success_rate[1-strategy]), lb, ub)  # Change: Adaptive scaling factor
                else:
                    b, c = pop[np.random.choice(idxs, 2, replace=False)]
                    mutant = np.clip(best_individual + randomized_F * (b - c) * np.random.uniform(0.9, 1.1), lb, ub)  # Change: Dynamic scaling factor
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial
                    self.success_rate[strategy] += 1
                    self.recent_success[strategy] += 1  # Change: Track recent success
                else:
                    new_pop.append(pop[i])
                    new_fitness.append(fitness[i])
            
            pop = np.array(new_pop)
            fitness = np.array(new_fitness)
            pop_size = max(2, int(self.pop_shrink_factor * len(pop)))
            
            self.F = np.clip(self.F + self.learning_rate * (np.random.rand() - 0.5), 0.1, 1.0)
            self.CR = np.clip(0.5 + 0.5 * (self.success_rate[1] / max(1, sum(self.success_rate))), 0.1, 1.0)
            
            total_recent_success = sum(self.recent_success)  # Change: Use recent success for strategy probability
            if total_recent_success > 0:
                self.strategy_prob = [(1 - self.learning_rate) * prob + self.learning_rate * (rate / total_recent_success) for prob, rate in zip(self.strategy_prob, self.recent_success)]
                self.recent_success = [0, 0]  # Change: Reset recent success
        
        return best_individual, best_fitness