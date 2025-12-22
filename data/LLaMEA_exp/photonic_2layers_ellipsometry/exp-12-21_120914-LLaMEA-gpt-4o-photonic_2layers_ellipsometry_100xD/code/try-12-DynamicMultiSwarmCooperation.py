import numpy as np

class DynamicMultiSwarmCooperation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.main_swarm_size = 10
        self.sub_swarm_size = 5
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        main_swarm = np.random.uniform(lb, ub, (self.main_swarm_size, self.dim))
        velocities = np.zeros((self.main_swarm_size, self.dim))
        personal_best = main_swarm.copy()
        personal_best_scores = np.array([func(ind) for ind in main_swarm])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        self.history.append(global_best_score)

        iter_count = 0
        inertia_weight = 0.9
        min_inertia = 0.4

        while iter_count < self.budget - self.main_swarm_size:
            inertia_weight = max(min_inertia, inertia_weight - 0.01)
            
            # Update main swarm
            cognitive_component = np.random.uniform(size=(self.main_swarm_size, self.dim)) * (personal_best - main_swarm)
            social_component = np.random.uniform(size=(self.main_swarm_size, self.dim)) * (global_best - main_swarm)
            velocities = inertia_weight * velocities + cognitive_component * np.random.rand() + social_component * np.random.rand()
            main_swarm += velocities

            main_swarm = np.clip(main_swarm, lb, ub)
            scores = np.array([func(ind) for ind in main_swarm])
            iter_count += self.main_swarm_size

            better_indices = scores < personal_best_scores
            personal_best[better_indices] = main_swarm[better_indices]
            personal_best_scores[better_indices] = scores[better_indices]

            min_index = np.argmin(personal_best_scores)
            if personal_best_scores[min_index] < global_best_score:
                global_best = personal_best[min_index]
                global_best_score = personal_best_scores[min_index]

            # Periodic knowledge sharing and sub-swarm evaluation
            if iter_count % (self.main_swarm_size * 5) == 0:
                sub_swarm = np.random.uniform(lb, ub, (self.sub_swarm_size, self.dim))
                sub_swarm_scores = np.array([func(ind) for ind in sub_swarm])
                iter_count += self.sub_swarm_size
                if np.min(sub_swarm_scores) < global_best_score:
                    global_best = sub_swarm[np.argmin(sub_swarm_scores)]
                    global_best_score = np.min(sub_swarm_scores)

                # Introduce small perturbations if no improvement
                if len(self.history) > 2 and self.history[-1] >= self.history[-2]:
                    perturbation = np.random.normal(0, 0.1, size=(self.main_swarm_size, self.dim))
                    main_swarm += perturbation

                self.history.append(global_best_score)

        return global_best_score, global_best