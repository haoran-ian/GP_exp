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
            # Adaptive inertia weight adjustment
            self.w = 0.3 + 0.3 * (self.budget - eval_count) / self.budget
            
            # Adaptive cognitive and social coefficients
            self.c1 = 1.0 + 0.5 * (eval_count / self.budget)  # Changed line 1
            self.c2 = 2.0 - 0.5 * (eval_count / self.budget)  # Changed line 2

            # PSO update with dynamic velocity bounds
            velocity_bound = 0.5 + 0.5 * (self.budget - eval_count) / self.budget
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = np.clip(
                self.w * velocities 
                + self.c1 * r1 * (pbest_positions - positions) 
                + self.c2 * r2 * (gbest_position - positions), 
                -velocity_bound, velocity_bound
            )
            positions = np.clip(positions + velocities, self.lb, self.ub)
            
            # Evaluate the new positions
            scores = np.array([func(x) for x in positions])
            eval_count += self.pop_size
            
            # Update personal best
            improved = scores < pbest_scores
            pbest_scores[improved] = scores[improved]
            pbest_positions[improved] = positions[improved]
            
            # Update global best
            gbest_idx = np.argmin(pbest_scores)
            if pbest_scores[gbest_idx] < gbest_score:
                gbest_position = pbest_positions[gbest_idx]
                gbest_score = pbest_scores[gbest_idx]

            # Dynamic scaling factor and crossover rate
            self.F = 0.6 + 0.3 * (self.budget - eval_count) / self.budget
            self.CR = 0.85 - 0.05 * (self.budget - eval_count) / self.budget

            # DE mutation and crossover
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b = np.random.choice(indices, 2, replace=False)
                # Introduce a new local search component
                if np.random.rand() < 0.25:  # Modified line 7
                    mutant_vector = np.clip(positions[i] + np.random.normal(0, 0.3, self.dim), self.lb, self.ub)  # Modified line 7
                else:
                    mutant_vector = np.clip(gbest_position + self.F * (positions[a] - positions[b]) + np.random.normal(0, 0.1, self.dim), self.lb, self.ub)  # Modified line 7
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):  # Ensure at least one dimension is taken from mutant_vector
                    crossover[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover, mutant_vector, positions[i])

                # Selection
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