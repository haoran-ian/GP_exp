import numpy as np

class Enhanced_APSO_DTA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.w = 0.9  # Initial inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.initial_temp = 1.0  # Initial temperature for annealing
        self.final_temp = 0.01  # Final temperature for annealing
        self.particles = np.random.rand(self.pop_size, dim)
        self.velocities = np.random.uniform(-0.1, 0.1, (self.pop_size, dim))
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.pop_size, np.inf)
        self.best_global_position = np.zeros(dim)
        self.best_global_score = np.inf
        self.func_eval_count = 0

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * (L**(-1/3))
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1/3)
        return step

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        
        self.particles = bounds_lb + self.particles * (bounds_ub - bounds_lb)
        
        while self.func_eval_count < self.budget:
            for i in range(self.pop_size):
                current_score = func(self.particles[i])
                self.func_eval_count += 1
                
                if current_score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = current_score
                    self.best_personal_positions[i] = self.particles[i].copy()
                
                if current_score < self.best_global_score:
                    self.best_global_score = current_score
                    self.best_global_position = self.particles[i].copy()
            
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            adapt_c1 = self.c1 * (1 - self.func_eval_count / self.budget)
            adapt_c2 = self.c2 * (self.func_eval_count / self.budget)
            personal_dist = np.linalg.norm(self.best_personal_positions - self.particles, axis=1, keepdims=True)
            global_dist = np.linalg.norm(self.best_global_position - self.particles, axis=1, keepdims=True)
            self.velocities = (self.w * self.velocities +
                               adapt_c1 * r1 * (self.best_personal_positions - self.particles) / (personal_dist + 1e-6) +
                               adapt_c2 * r2 * (self.best_global_position - self.particles) / (global_dist + 1e-6))
            self.particles += self.velocities
            
            temp = self.initial_temp * ((self.final_temp / self.initial_temp) ** (self.func_eval_count / self.budget))
            perturbation = np.random.normal(0, temp, (self.pop_size, self.dim))
            self.particles += perturbation
            
            # Incorporate Levy Flight for additional exploration
            if np.random.rand() < 0.3:  # 30% chance to perform Levy Flight
                L = np.random.uniform(0.1, 0.5)
                levy_steps = np.array([self.levy_flight(L) for _ in range(self.pop_size)])
                self.particles += levy_steps
            
            # Dynamic inertia weight decay
            self.w = self.w_min + (0.5 * (1 - self.func_eval_count / self.budget) * (self.w - self.w_min))
            
            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)
        
        return self.best_global_position, self.best_global_score