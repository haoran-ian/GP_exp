import numpy as np

class Adaptive_Diversity_QHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hms = min(50, budget // 10)  # Harmony memory size
        self.hmcr_init = 0.95  # Initial Harmony memory consideration rate
        self.hmcr_final = 0.65  # Final Harmony memory consideration rate
        self.par_init = 0.5  # Initial Pitch adjustment rate
        self.par_final = 0.05  # Final Pitch adjustment rate
        self.evaluations = 0
        self.quantum_prob_init = 0.25  # Initial Quantum probability
        self.learning_rate_init = 0.2  # Initial Adaptive learning rate
        self.learning_rate_final = 0.05  # Final Adaptive learning rate

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
            quantum_prob = self.quantum_prob_init * (1 - progress) + 0.1 * progress
            learning_rate = self.learning_rate_init - (self.learning_rate_init - self.learning_rate_final) * progress

            new_harmony = np.array([harmony_memory[np.random.randint(self.hms), j] 
                                    if np.random.rand() < hmcr else np.random.uniform(lb[j], ub[j]) 
                                    for j in range(self.dim)])

            if np.random.rand() < par:
                gradient_approximation = (ub - lb) / self.dim * (-1 + 2 * np.random.rand(self.dim))
                new_harmony += learning_rate * gradient_approximation
                new_harmony = np.clip(new_harmony, lb, ub)

            if np.random.rand() < quantum_prob:
                center = (np.mean(harmony_memory, axis=0) + best_harmony) / 2
                delta = np.abs(best_harmony - harmony_memory[np.random.randint(self.hms)])
                new_harmony = center + np.random.uniform(-1, 1, self.dim) * delta / np.sqrt(2)
                new_harmony = np.clip(new_harmony, lb, ub)

            score = func(new_harmony)
            self.evaluations += 1

            if score < best_score:
                best_score = score
                best_harmony = new_harmony

            worst_idx = np.argmax(harmony_scores)
            if score < harmony_scores[worst_idx]:
                harmony_memory[worst_idx], harmony_scores[worst_idx] = new_harmony, score

        return best_harmony, best_score