import numpy as np

class Enhanced_QHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hms = min(50, budget // 10)  
        self.hmcr_init = 0.95  
        self.hmcr_final = 0.65  # Change: Adjusted final harmony memory consideration rate
        self.par_init = 0.5  
        self.par_final = 0.10  # Change: Adjusted final pitch adjustment rate
        self.evaluations = 0
        self.quantum_prob = 0.25  # Change: Increased initial quantum probability
        self.learning_rate = 0.12  # Change: Adjusted adaptive learning rate

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
            adaptive_quantum_prob = self.quantum_prob * (1 - progress) + 0.25 * progress  # Change: Minor adjustment

            new_harmony = np.array([harmony_memory[np.random.randint(self.hms), j] 
                                    if np.random.rand() < hmcr else np.random.uniform(lb[j], ub[j]) 
                                    for j in range(self.dim)])

            if np.random.rand() < par:
                gradient_approximation = (ub - lb) / self.dim * (-1 + 2 * np.random.rand(self.dim))
                new_harmony += self.learning_rate * gradient_approximation
                new_harmony = np.clip(new_harmony, lb, ub)

            if np.random.rand() < adaptive_quantum_prob:
                center = (np.mean(harmony_memory, axis=0) + best_harmony) / 2
                delta = np.abs(best_harmony - harmony_memory[np.random.randint(self.hms)])
                dynamic_delta_factor = np.sqrt(2) * (1 + progress)
                new_harmony = center + np.random.uniform(-1, 1, self.dim) * delta / dynamic_delta_factor
                new_harmony = np.clip(new_harmony, lb, ub)

            score = func(new_harmony)
            self.evaluations += 1

            if score < best_score:
                best_score = score
                best_harmony = new_harmony
                self.learning_rate = max(0.03, self.learning_rate * 0.97)  # Change: Reduce learning rate if improvement

            worst_idx = np.argmax(harmony_scores)
            if score < harmony_scores[worst_idx]:
                harmony_memory = np.delete(harmony_memory, worst_idx, axis=0)  # Change: Remove worst harmony
                harmony_scores = np.delete(harmony_scores, worst_idx)  # Change: Update scores
                harmony_memory = np.vstack([harmony_memory, new_harmony])  # Change: Add new harmony
                harmony_scores = np.append(harmony_scores, score)  # Change: Append new score

        return best_harmony, best_score