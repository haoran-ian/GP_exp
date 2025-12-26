import numpy as np

class EnhancedHybridPSO_DE_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.F_max = 1.0
        self.F_min = 0.5
        self.CR = 0.9
        self.epsilon = 1e-8
        # Initialize feedback mechanism
        self.improvement_threshold = 0.01
        self.recent_improvements = []
        self.feedback_window = 5
        # Chaotic map initialization
        self.chaos_factor = np.random.rand()

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = func(global_best_position)
        
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Calculate diversity
            diversity = np.mean(np.std(swarm, axis=0))
            # Adapt inertia weight based on diversity
            self.w = self.w_max - (self.w_max - self.w_min) * (diversity / (diversity + self.epsilon))
            # Chaotic map update
            self.chaos_factor = (4 * self.chaos_factor) * (1 - self.chaos_factor)
            # Self-adaptive strategy for F and CR
            self.F = self.F_min + self.chaos_factor * (self.F_max - self.F_min) 
            self.CR = 0.6 + 0.35 * self.chaos_factor  # Dynamic CR adjustment
            
            # PSO Update
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - swarm) +
                          self.c2 * r2 * (global_best_position - swarm))
            swarm = np.clip(swarm + velocities, lower_bound, upper_bound)
            
            # DE Update
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
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
                    if trial_score < global_best_score:
                        # Record improvement
                        self.recent_improvements.append(global_best_score - trial_score)
                        global_best_position = trial
                        global_best_score = trial_score
                        # Limit feedback window size
                        if len(self.recent_improvements) > self.feedback_window:
                            self.recent_improvements.pop(0)

            # Feedback mechanism: Increase exploration if no improvements
            if len(self.recent_improvements) >= self.feedback_window and np.mean(self.recent_improvements) < self.improvement_threshold:
                self.c1 = np.clip(self.c1 + 0.1, 0, 2.0)
                self.c2 = np.clip(self.c2 - 0.1, 0, 2.0)
                self.recent_improvements = []

            if eval_count >= self.budget:
                break
        
        return global_best_position