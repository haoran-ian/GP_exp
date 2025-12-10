import numpy as np

class HPSO_QAE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.w_max = 0.9  # Max inertia weight
        self.w_min = 0.4  # Min inertia weight
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.quantum_exploration_factor = 0.05  # Quantum exploration factor
        self.particles = np.random.rand(self.pop_size, dim)
        self.velocities = np.random.uniform(-0.1, 0.1, (self.pop_size, dim))
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.pop_size, np.inf)
        self.best_global_position = np.zeros(dim)
        self.best_global_score = np.inf
        self.func_eval_count = 0

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        
        # Normalize particles to the function's bounds
        self.particles = bounds_lb + self.particles * (bounds_ub - bounds_lb)
        
        while self.func_eval_count < self.budget:
            for i in range(self.pop_size):
                current_score = func(self.particles[i])
                self.func_eval_count += 1
                
                # Update personal best
                if current_score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = current_score
                    self.best_personal_positions[i] = self.particles[i].copy()
                
                # Update global best
                if current_score < self.best_global_score:
                    self.best_global_score = current_score
                    self.best_global_position = self.particles[i].copy()
            
            # Calculate adaptive inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (self.func_eval_count / self.budget)
            
            # Update velocities and positions
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.velocities = (w * self.velocities +
                               self.c1 * r1 * (self.best_personal_positions - self.particles) +
                               self.c2 * r2 * (self.best_global_position - self.particles))
            self.particles += self.velocities
            
            # Quantum-inspired exploration mechanism
            quantum_jump = np.random.normal(0, self.quantum_exploration_factor, (self.pop_size, self.dim))
            exploration_mask = np.random.rand(self.pop_size, self.dim) < self.quantum_exploration_factor
            self.particles += exploration_mask * quantum_jump
            
            # Keep particles inside bounds
            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)
        
        return self.best_global_position, self.best_global_score