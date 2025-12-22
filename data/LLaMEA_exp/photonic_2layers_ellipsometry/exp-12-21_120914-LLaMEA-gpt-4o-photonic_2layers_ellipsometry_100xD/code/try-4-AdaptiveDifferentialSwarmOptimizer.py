import numpy as np

class AdaptiveDifferentialSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initial swarm setup
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
        crossover_rate = 0.9
        differential_weight = 0.8

        while iter_count < self.budget - self.population_size:
            # Dynamic inertia weight adaptation
            inertia_weight = max(min_inertia, inertia_weight - 0.01)
            
            # Update velocities and positions using swarm and differential evolution principles
            for i in range(self.population_size):
                indices = np.arange(self.population_size)
                np.random.shuffle(indices)
                a, b, c = swarm[indices[:3]]
                
                # Differential mutation
                mutant_vector = a + differential_weight * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                # Crossover
                trial_vector = np.where(np.random.rand(self.dim) < crossover_rate, mutant_vector, swarm[i])
                trial_vector = np.clip(trial_vector, lb, ub)
                
                trial_score = func(trial_vector)
                iter_count += 1
                
                # Selection
                if trial_score < personal_best_scores[i]:
                    personal_best[i] = trial_vector
                    personal_best_scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best = trial_vector
                        global_best_score = trial_score

                # Update velocities and positions based on personal and global bests
                cognitive_component = np.random.uniform(size=self.dim) * (personal_best[i] - swarm[i])
                social_component = np.random.uniform(size=self.dim) * (global_best - swarm[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], lb, ub)
            
            # Adaptive perturbation
            if iter_count % (self.population_size * 5) == 0:
                if len(self.history) > 2 and self.history[-1] >= self.history[-2]:
                    perturbation = np.random.normal(0, 0.1, size=(self.population_size, self.dim))
                    swarm += perturbation
                self.history.append(global_best_score)
        
        return global_best_score, global_best