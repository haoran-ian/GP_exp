import numpy as np

class EnhancedAdvancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = 50
        num_swarms = 2  # Number of sub-swarms for dynamic multi-swarm strategy
        swarm_size = population_size // num_swarms
        swarms = [np.random.uniform(lb, ub, (swarm_size, self.dim)) for _ in range(num_swarms)]
        velocities = [np.random.uniform(-1, 1, (swarm_size, self.dim)) for _ in range(num_swarms)]
        personal_best_positions = [np.copy(swarm) for swarm in swarms]
        personal_best_scores = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        global_best_indices = [np.argmin(scores) for scores in personal_best_scores]
        global_best_positions = [np.copy(personal_best_positions[i][global_best_indices[i]]) for i in range(num_swarms)]
        global_best_scores = [personal_best_scores[i][global_best_indices[i]] for i in range(num_swarms)]
        
        eval_count = population_size

        # Adaptive inertia weight parameters
        w_max = 0.9
        w_min = 0.4
        c1, c2 = 1.5, 1.5
        F_base = 0.8  # Base DE scaling factor
        CR = 0.9  # Crossover probability

        while eval_count < self.budget:
            for s in range(num_swarms):
                # Adaptive inertia weight
                w = w_max - ((w_max - w_min) * (eval_count / self.budget))
                
                # Particle Swarm Optimization update
                r1, r2 = np.random.rand(2)
                velocities[s] = (w * velocities[s] + 
                                 c1 * r1 * (personal_best_positions[s] - swarms[s]) + 
                                 c2 * r2 * (global_best_positions[s] - swarms[s]))
                swarms[s] += velocities[s]
                swarms[s] = np.clip(swarms[s], lb, ub)

                # Differential Evolution mutation and crossover with adaptive strategy
                for i in range(swarm_size):
                    candidates = list(range(swarm_size))
                    candidates.remove(i)
                    a, b, c = np.random.choice(candidates, 3, replace=False)
                
                    # Self-adaptive mutation strategy with learning factor
                    F = F_base + 0.2 * np.random.rand()
                    learning_factor = 0.5 + 0.5 * np.random.rand()
                    mutant = np.clip(swarms[s][a] + F * (swarms[s][b] - swarms[s][c]) * learning_factor, lb, ub)
                    
                    crossover = np.random.rand(self.dim) < CR
                    trial = np.where(crossover, mutant, swarms[s][i])

                    f_trial = func(trial)
                    eval_count += 1

                    if f_trial < personal_best_scores[s][i]:
                        personal_best_positions[s][i] = trial
                        personal_best_scores[s][i] = f_trial
                        if f_trial < global_best_scores[s]:
                            global_best_positions[s] = trial
                            global_best_scores[s] = f_trial
                            # Neighborhood-based elitism
                            neighbors = np.argsort(personal_best_scores[s])[:3]
                            for neighbor in neighbors:
                                if personal_best_scores[s][neighbor] > f_trial:
                                    personal_best_positions[s][neighbor] = trial
                                    personal_best_scores[s][neighbor] = f_trial

                    if eval_count >= self.budget:
                        break

        overall_best_index = np.argmin(global_best_scores)
        return {'best_position': global_best_positions[overall_best_index], 'best_score': global_best_scores[overall_best_index]}