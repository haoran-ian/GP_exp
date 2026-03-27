import numpy as np

class Enhanced_QHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hms = min(50, budget // 10)
        self.hmcr_init = 0.95
        self.hmcr_final = 0.70
        self.par_init = 0.5
        self.par_final = 0.05
        self.evaluations = 0
        self.quantum_prob = 0.20
        self.learning_rate = 0.15

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.hms, self.dim))
        harmony_scores = np.array([func(harmony_memory[i]) for i in range(self.hms)])
        self.evaluations += self.hms

        best_harmony = harmony_memory[np.argmin(harmony_scores)]
        best_score = np.min(harmony_scores)

        while self.evaluations < self.budget:
            progress = self.evaluations / self.budget
            hmcr = self.hmcr_init - (self.hmcr_init - self.hmcr_final) * progress
            par = self.par_init - (self.par_init - self.par_final) * progress
            adaptive_quantum_prob = self.quantum_prob * (1 - progress) + 0.20 * progress

            new_harmony = np.array([harmony_memory[np.random.randint(self.hms), j] 
                                    if np.random.rand() < hmcr else np.random.uniform(lb[j], ub[j]) 
                                    for j in range(self.dim)])

            if np.random.rand() < par:
                dynamic_step = (1 - progress) * 0.1
                gradient_approximation = (ub - lb) / self.dim * (-1 + 2 * np.random.rand(self.dim))
                new_harmony += self.learning_rate * gradient_approximation * dynamic_step
                new_harmony = np.clip(new_harmony, lb, ub)

            if np.random.rand() < adaptive_quantum_prob:
                center = (np.mean(harmony_memory, axis=0) + best_harmony) / 2
                delta = np.abs(best_harmony - harmony_memory[np.random.randint(self.hms)])
                dynamic_delta_factor = np.sqrt(1.5) * (1 + progress)  # Adjusted dynamic delta factor
                new_harmony = center + np.random.uniform(-1, 1, self.dim) * delta / dynamic_delta_factor
                new_harmony = np.clip(new_harmony, lb, ub)

            score = func(new_harmony)
            self.evaluations += 1

            if score < best_score:
                best_score = score
                best_harmony = new_harmony
                self.learning_rate = max(0.05, self.learning_rate * 0.98)

            worst_idx = np.argmax(harmony_scores)
            adaptive_memory_update_threshold = 0.1 + 0.5 * progress  # New memory update threshold
            if score < harmony_scores[worst_idx] or np.random.rand() < adaptive_memory_update_threshold:
                harmony_memory[worst_idx], harmony_scores[worst_idx] = new_harmony, score

        return best_harmony, best_score