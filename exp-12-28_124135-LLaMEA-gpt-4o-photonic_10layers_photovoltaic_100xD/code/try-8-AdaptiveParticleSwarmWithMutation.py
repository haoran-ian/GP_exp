import numpy as np

class AdaptiveParticleSwarmWithMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget)
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_factor = 1.5
        self.social_factor = 1.5
        self.vel_range_factor = 0.1
        
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
            
            better_mask = scores < personal_best_scores
            personal_best_scores = np.where(better_mask, scores, personal_best_scores)
            personal_best_positions = np.where(better_mask[:, np.newaxis], positions, personal_best_positions)
            
            min_score = np.min(scores)
            if min_score < global_best_score:
                global_best_score = min_score
                global_best_position = positions[np.argmin(scores)]
            
            # Adaptive inertia weight
            inertia_weight = self.inertia_weight_final + \
                             (self.inertia_weight_initial - self.inertia_weight_final) * \
                             ((self.budget - evaluations) / self.budget)
            
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = (
                inertia_weight * velocities +
                self.cognitive_factor * r1 * (personal_best_positions - positions) +
                self.social_factor * r2 * (global_best_position - positions)
            )
            
            velocities = np.clip(velocities, -velocity_range, velocity_range)
            positions = positions + velocities
            positions = np.clip(positions, lb, ub)
            
            # Non-Uniform Mutation
            if evaluations % (self.budget // 10) == 0:
                mutation_rate = 1 - (evaluations / self.budget)
                mutation_scale = mutation_rate * velocity_range
                mutation = np.random.normal(0, mutation_scale, positions.shape)
                mutated_positions = positions + mutation
                mutated_positions = np.clip(mutated_positions, lb, ub)
                mutated_scores = np.array([func(pos) for pos in mutated_positions])
                evaluations += self.swarm_size
                improvement_mask = mutated_scores < scores
                positions = np.where(improvement_mask[:, np.newaxis], mutated_positions, positions)

        return global_best_position, global_best_score