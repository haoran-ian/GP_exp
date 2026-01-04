import numpy as np

class AdvancedAdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget)
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_factor = 1.5
        self.social_factor = 1.5
        self.vel_range_factor_initial = 0.1
        self.vel_range_factor_final = 0.05
        
    def __call__(self, func):
        lb = np.array(func[0].bounds.lb)
        ub = np.array(func[0].bounds.ub)
        search_range = ub - lb
        
        positions = lb + np.random.rand(self.swarm_size, self.dim) * search_range
        velocities = np.random.uniform(-search_range, search_range, (self.swarm_size, self.dim)) * self.vel_range_factor_initial
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        
        global_best_score = np.inf
        global_best_position = None
        
        evaluations = 0
        while evaluations < self.budget:
            scores = np.array([func[0](pos) for pos in positions])
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
                             
            # Adaptive velocity range factor
            vel_range_factor = self.vel_range_factor_final + \
                               (self.vel_range_factor_initial - self.vel_range_factor_final) * \
                               ((self.budget - evaluations) / self.budget)
            velocity_range = vel_range_factor * search_range
            
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
            
            # Dynamic personal best refinement
            refinement_mask = np.random.rand(self.swarm_size) < 0.1
            refined_positions = positions + np.random.uniform(-0.01, 0.01, (self.swarm_size, self.dim)) * search_range
            refined_scores = np.array([func[0](pos) for pos in refined_positions])
            evaluations += self.swarm_size
            
            better_refinement_mask = refined_scores < scores
            positions = np.where(better_refinement_mask[:, np.newaxis], refined_positions, positions)
            
            # Update personal bests after refinement
            refined_better_mask = refined_scores < personal_best_scores
            personal_best_scores = np.where(refined_better_mask, refined_scores, personal_best_scores)
            personal_best_positions = np.where(refined_better_mask[:, np.newaxis], refined_positions, personal_best_positions)
            
            if evaluations >= self.budget:
                break
            
        return global_best_position, global_best_score