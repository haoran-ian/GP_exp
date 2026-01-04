import numpy as np

class DynamicParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget)  # Limit swarm size to be small enough given the budget
        self.inertia_weight = 0.7
        self.cognitive_factor = 1.5
        self.social_factor = 1.5
        self.vel_range_factor = 0.1  # Initial range for velocities
        
    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        search_range = ub - lb
        velocity_range = self.vel_range_factor * search_range
        
        positions = lb + np.random.rand(self.swarm_size, self.dim) * search_range
        velocities = np.random.uniform(-velocity_range, velocity_range, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        
        global_best_score = np.inf
        global_best_position = None
        
        evaluations = 0
        
        while evaluations < self.budget:
            scores = np.array([func(pos) for pos in positions])
            evaluations += self.swarm_size
            
            # Update personal bests
            better_mask = scores < personal_best_scores
            personal_best_scores = np.where(better_mask, scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, np.newaxis], positions, personal_best_positions)
            
            # Update global best
            min_score = np.min(scores)
            if min_score < global_best_score:
                global_best_score = min_score
                global_best_position = positions[np.argmin(scores)]
            
            # Update velocities and positions
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = (
                self.inertia_weight * velocities +
                self.cognitive_factor * r1 * (personal_best_positions - positions) +
                self.social_factor * r2 * (global_best_position - positions)
            )
            
            # Dynamic boundary adaptation for velocities
            velocities = np.clip(velocities, -velocity_range, velocity_range)
            
            # Reinitialize velocity if stagnant
            if evaluations % (self.budget // 5) == 0:  # Reinitialize periodically
                stagnation_index = np.random.choice(self.swarm_size, int(self.swarm_size * 0.1), replace=False)
                velocities[stagnation_index] = np.random.uniform(-velocity_range, velocity_range, (len(stagnation_index), self.dim))
            
            positions = positions + velocities
            positions = np.clip(positions, lb, ub)  # Ensure the particles stay within bounds
        
        return global_best_position, global_best_score