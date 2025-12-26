import numpy as np

class MultiSwarmEnhancedPerturbation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.num_swarms = 3  # New line
        self.history = []
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarms = [np.random.uniform(lb, ub, (self.initial_population_size, self.dim)) for _ in range(self.num_swarms)]  # Modified line
        velocities = [np.zeros((self.initial_population_size, self.dim)) for _ in range(self.num_swarms)]  # Modified line
        
        personal_bests, personal_best_scores, global_best, global_best_score = [], [], None, float('inf')  # Modified line
        
        for swarm in swarms:  # New block
            p_best = swarm.copy()
            p_best_scores = np.array([func(ind) for ind in swarm])
            personal_bests.append(p_best)
            personal_best_scores.append(p_best_scores)
        
            best_index = np.argmin(p_best_scores)
            if p_best_scores[best_index] < global_best_score:
                global_best = p_best[best_index]
                global_best_score = p_best_scores[best_index]
        
        self.history.append(global_best_score)

        iter_count = 0
        inertia_weight = 0.9
        min_inertia = 0.4
        
        while iter_count < self.budget - self.initial_population_size * self.num_swarms:  # Modified line
            inertia_weight = max(min_inertia, inertia_weight - 0.01)
            
            for i, swarm in enumerate(swarms):
                cognitive_component = np.random.uniform(size=(self.initial_population_size, self.dim)) * (personal_bests[i] - swarm)
                social_component = np.random.uniform(size=(self.initial_population_size, self.dim)) * (global_best - swarm)
                
                velocities[i] = inertia_weight * velocities[i] + cognitive_component * np.random.rand() + social_component * np.random.rand()
                swarm += velocities[i]

                swarm = np.clip(swarm, lb, ub)

                scores = np.array([func(ind) for ind in swarm])
                iter_count += self.initial_population_size

                better_indices = scores < personal_best_scores[i]
                personal_bests[i][better_indices] = swarm[better_indices]
                personal_best_scores[i][better_indices] = scores[better_indices]

                min_index = np.argmin(personal_best_scores[i])
                if personal_best_scores[i][min_index] < global_best_score:
                    global_best = personal_bests[i][min_index]
                    global_best_score = personal_best_scores[i][min_index]

            if iter_count % (self.initial_population_size * self.num_swarms * 5) == 0:
                if len(self.history) > 2 and self.history[-1] >= self.history[-2]:
                    perturbation = np.random.normal(0, 0.1, size=(self.initial_population_size, self.dim))
                    for swarm in swarms:
                        swarm += perturbation
                self.history.append(global_best_score)

            if iter_count % (self.initial_population_size * self.num_swarms * 10) == 0:  # New line
                swarms.append(np.random.uniform(lb, ub, (self.initial_population_size, self.dim)))  # New line

        return global_best_score, global_best