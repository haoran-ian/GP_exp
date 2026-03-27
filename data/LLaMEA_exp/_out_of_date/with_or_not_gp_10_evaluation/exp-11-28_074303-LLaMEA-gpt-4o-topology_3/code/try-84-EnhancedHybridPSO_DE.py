import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 10 + 2 * dim
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.6
        self.F = 0.8
        self.CR = 0.9
        self.diversity_threshold = 0.1
        self.extra_mutation_prob = 0.1

    def __call__(self, func):
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        pbest_positions = np.copy(positions)
        pbest_scores = np.array([func(x) for x in pbest_positions])
        
        gbest_idx = np.argmin(pbest_scores)
        gbest_position = pbest_positions[gbest_idx]
        gbest_score = pbest_scores[gbest_idx]

        eval_count = self.pop_size

        while eval_count < self.budget:
            self.w = 0.3 + 0.3 * (np.std(positions) / (self.ub - self.lb))  # Diversity-based adaptive inertia weight
            self.c1 = 1.0 + 0.5 * (eval_count / self.budget)
            self.c2 = 2.0 - 0.5 * (eval_count / self.budget)

            # Changed the lower limit of velocity_bound for improved convergence
            velocity_bound = 0.3 + 0.7 * (self.budget - eval_count) / self.budget  # Adjusted velocity clamping
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = np.clip(
                self.w * velocities 
                + self.c1 * r1 * (pbest_positions - positions) 
                + self.c2 * r2 * (gbest_position - positions), 
                -velocity_bound, velocity_bound
            )
            positions = np.clip(positions + velocities, self.lb, self.ub)
            
            scores = np.array([func(x) for x in positions])
            eval_count += self.pop_size
            
            improved = scores < pbest_scores
            pbest_scores[improved] = scores[improved]
            pbest_positions[improved] = positions[improved]
            
            gbest_idx = np.argmin(pbest_scores)
            if pbest_scores[gbest_idx] < gbest_score:
                gbest_position = pbest_positions[gbest_idx]
                gbest_score = pbest_scores[gbest_idx]

            self.F = 0.5 + 0.4 * np.random.rand()  # Adaptive mutation scaling
            self.CR = 0.85 - 0.05 * (self.budget - eval_count) / self.budget

            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b = np.random.choice(indices, 2, replace=False)
                if np.random.rand() < 0.2:
                    mutant_vector = np.clip(positions[i] + np.random.normal(0, 0.1, self.dim), self.lb, self.ub)
                else:
                    mutant_vector = np.clip(gbest_position + self.F * (positions[a] - positions[b]), self.lb, self.ub)
                
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover, mutant_vector, positions[i])

                if np.random.rand() < self.extra_mutation_prob + 0.05 * (eval_count / self.budget):
                    trial_vector += np.random.normal(0, self.diversity_threshold, self.dim)

                trial_score = func(trial_vector)
                eval_count += 1
                if trial_score < scores[i]:
                    positions[i] = trial_vector
                    scores[i] = trial_score
                    if trial_score < pbest_scores[i]:
                        pbest_scores[i] = trial_score
                        pbest_positions[i] = trial_vector
                        if trial_score < gbest_score:
                            gbest_position = trial_vector
                            gbest_score = trial_score

        return gbest_position, gbest_score