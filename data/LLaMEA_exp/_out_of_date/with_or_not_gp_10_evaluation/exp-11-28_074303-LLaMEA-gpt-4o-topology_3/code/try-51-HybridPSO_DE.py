import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 10 + 2 * dim  # population size
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.6   # inertia weight
        self.F = 0.8   # DE scaling factor
        self.CR = 0.9  # DE crossover rate

    def __call__(self, func):
        def crowding_distance(positions):
            sorted_idx = np.argsort(positions, axis=0)
            distances = np.zeros_like(positions)
            distances[sorted_idx[0]] = np.inf
            max_min_diff = np.max(positions, axis=0) - np.min(positions, axis=0)
            for i in range(1, len(positions) - 1):
                distances[sorted_idx[i]] += np.sum(
                    (positions[sorted_idx[i+1]] - positions[sorted_idx[i-1]]) / max_min_diff
                )
            distances[sorted_idx[-1]] = np.inf
            return distances
        
        # Initialize positions and velocities for PSO
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        pbest_positions = np.copy(positions)
        pbest_scores = np.array([func(x) for x in pbest_positions])
        
        gbest_idx = np.argmin(pbest_scores)
        gbest_position = pbest_positions[gbest_idx]
        gbest_score = pbest_scores[gbest_idx]

        eval_count = self.pop_size

        while eval_count < self.budget:
            self.w = 0.3 + 0.3 * (self.budget - eval_count) / self.budget
            velocity_bound = 0.5 + 0.5 * (self.budget - eval_count) / self.budget
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

            self.F = 0.6 + 0.3 * (self.budget - eval_count) / self.budget
            self.CR = 0.85 - 0.05 * (self.budget - eval_count) / self.budget

            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = np.clip(positions[a] + self.F * (positions[b] - positions[c]), self.lb, self.ub)
                
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover, mutant_vector, positions[i])

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < scores[i] or (trial_score == scores[i] and crowding_distance([positions[i], trial_vector])[1] > crowding_distance([positions[i], trial_vector])[0]):
                    positions[i] = trial_vector
                    scores[i] = trial_score
                    if trial_score < pbest_scores[i]:
                        pbest_scores[i] = trial_score
                        pbest_positions[i] = trial_vector
                        if trial_score < gbest_score:
                            gbest_position = trial_vector
                            gbest_score = trial_score

        return gbest_position, gbest_score