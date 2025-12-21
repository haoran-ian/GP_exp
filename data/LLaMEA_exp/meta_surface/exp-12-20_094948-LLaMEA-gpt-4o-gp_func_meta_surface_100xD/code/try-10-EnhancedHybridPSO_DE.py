import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.min_pop_size = 10
        self.max_pop_size = 40
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.F_max = 1.0
        self.F_min = 0.5
        self.CR = 0.9
        self.epsilon = 1e-8

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_pop_size
        swarm = np.random.uniform(lower_bound, upper_bound, (pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (pop_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        eval_count = pop_size

        while eval_count < self.budget:
            # Calculate diversity
            diversity = np.mean(np.std(swarm, axis=0))
            # Adaptive population size based on budget usage
            pop_size = self.min_pop_size + int((self.max_pop_size - self.min_pop_size) * (eval_count / self.budget))
            swarm = np.resize(swarm, (pop_size, self.dim))
            velocities = np.resize(velocities, (pop_size, self.dim))
            # Adapt inertia weight based on diversity
            self.w = self.w_max - (self.w_max - self.w_min) * (diversity / (diversity + self.epsilon))
            
            # PSO Update
            r1, r2 = np.random.rand(pop_size, self.dim), np.random.rand(pop_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - swarm) +
                          self.c2 * r2 * (global_best_position - swarm))
            swarm = np.clip(swarm + velocities, lower_bound, upper_bound)
            
            # DE Update
            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                # Adaptive F based on diversity
                self.F = self.F_max - (self.F_max - self.F_min) * (diversity / (diversity + self.epsilon))
                mutant = np.clip(swarm[a] + self.F * (swarm[b] - swarm[c]), lower_bound, upper_bound)
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, swarm[i])
                trial_score = func(trial)
                eval_count += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score
                    if trial_score < func(global_best_position):
                        global_best_position = trial
            if eval_count >= self.budget:
                break
        
        return global_best_position