import numpy as np

class QuantumInspiredSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initial swarm setup
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        self.history.append(global_best_score)
        
        iter_count = 0
        amplitude = 1.0
        min_amplitude = 0.4

        while iter_count < self.budget - self.population_size:
            # Quantum-inspired probabilistic position update
            angle = np.random.uniform(-np.pi, np.pi, (self.population_size, self.dim))
            delta = amplitude * np.tan(angle)
            swarm += delta
            
            # Clip positions to bounds
            swarm = np.clip(swarm, lb, ub)
            
            # Evaluate new solutions
            scores = np.array([func(ind) for ind in swarm])
            iter_count += self.population_size
            
            # Update personal bests
            better_indices = scores < personal_best_scores
            personal_best[better_indices] = swarm[better_indices]
            personal_best_scores[better_indices] = scores[better_indices]
            
            # Update global best
            min_index = np.argmin(personal_best_scores)
            if personal_best_scores[min_index] < global_best_score:
                global_best = personal_best[min_index]
                global_best_score = personal_best_scores[min_index]
                
            # Adaptive amplitude control based on success history
            if iter_count % (self.population_size * 5) == 0:
                if len(self.history) > 2 and self.history[-1] >= self.history[-2]:
                    amplitude = max(min_amplitude, amplitude - 0.05)
                else:
                    amplitude = min(1.0, amplitude + 0.05)
                self.history.append(global_best_score)
        
        return global_best_score, global_best