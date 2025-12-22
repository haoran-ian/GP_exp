import numpy as np

class HybridizedSwarmDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        self.history.append(global_best_score)

        iter_count = 0
        inertia_weight = 0.9
        min_inertia = 0.4
        F = 0.5
        CR = 0.9

        while iter_count < self.budget - self.population_size:
            inertia_weight = max(min_inertia, inertia_weight - 0.01)
            
            cognitive_component = np.random.uniform(size=(self.population_size, self.dim)) * (personal_best - swarm)
            social_component = np.random.uniform(size=(self.population_size, self.dim)) * (global_best - swarm)
            
            velocities = inertia_weight * velocities + cognitive_component * np.random.rand() + social_component * np.random.rand()
            swarm += velocities

            # Apply Differential Evolution strategy
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = swarm[indices]
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, swarm[i])
                trial_score = func(trial)
                if trial_score < personal_best_scores[i]:
                    swarm[i] = trial
                    personal_best[i] = trial
                    personal_best_scores[i] = trial_score

            swarm = np.clip(swarm, lb, ub)

            scores = np.array([func(ind) for ind in swarm])
            iter_count += self.population_size

            better_indices = scores < personal_best_scores
            personal_best[better_indices] = swarm[better_indices]
            personal_best_scores[better_indices] = scores[better_indices]

            min_index = np.argmin(personal_best_scores)
            if personal_best_scores[min_index] < global_best_score:
                global_best = personal_best[min_index]
                global_best_score = personal_best_scores[min_index]

            # Dynamic population size adjustment
            if iter_count % (self.population_size * 5) == 0:
                self.population_size = min(self.population_size + 1, self.budget // 10)

            if iter_count % (self.population_size * 5) == 0:
                if len(self.history) > 2 and self.history[-1] >= self.history[-2]:
                    perturbation = np.random.normal(0, 0.1, size=(self.population_size, self.dim))
                    swarm += perturbation
                self.history.append(global_best_score)
        
        return global_best_score, global_best