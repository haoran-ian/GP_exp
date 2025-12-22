import numpy as np

class HybridQPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + int(2 * np.sqrt(dim))
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.beta = 0.5  # contraction-expansion coefficient for QPSO
        self.global_best_value = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        personal_best = np.copy(swarm)
        personal_best_values = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_values)]
        self.global_best_value = np.min(personal_best_values)
        
        eval_count = self.pop_size
        
        while eval_count < self.budget:
            mean_best = np.mean(personal_best, axis=0)
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                phi = np.random.uniform(-1, 1, self.dim)
                u = np.random.rand(self.dim)
                
                # QPSO update
                p = (personal_best[i] + global_best) / 2
                L = self.beta * np.log(1 / u)
                new_position = p + phi * np.abs(mean_best - swarm[i]) * L
                new_position = np.clip(new_position, lb, ub)

                # DE mutation and crossover
                if np.random.rand() < self.crossover_rate:
                    j_rand = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if np.random.rand() > self.crossover_rate and j != j_rand:
                            new_position[j] = swarm[np.random.randint(self.pop_size)][j]
                
                new_position_value = func(new_position)
                eval_count += 1
                
                if new_position_value < personal_best_values[i]:
                    personal_best[i] = new_position
                    personal_best_values[i] = new_position_value
                    if new_position_value < self.global_best_value:
                        global_best = new_position
                        self.global_best_value = new_position_value
        
        return global_best