import numpy as np

class Enhanced_QHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hms = min(50, budget // 10)  
        self.hmcr_init = 0.95  
        self.hmcr_final = 0.60  # Change 1: Adjusted final HMCR for better exploration
        self.par_init = 0.5  
        self.par_final = 0.02  # Change 2: Adjusted final PAR for more refined pitch adjustment
        self.evaluations = 0
        self.quantum_prob = 0.25  # Change 3: Increased initial Quantum probability for enhanced exploration
        self.learning_rate = 0.18  # Change 4: Adjusted learning rate for better convergence control

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
            adaptive_quantum_prob = self.quantum_prob * (1 - progress) + 0.25 * progress  # Change 5: Improved quantum adaptation

            dynamic_memory_factor = int(self.hms * (0.5 + 0.5 * progress))  # Change 6: Dynamic memory allocation

            new_harmony = np.array([harmony_memory[np.random.randint(dynamic_memory_factor), j] 
                                    if np.random.rand() < hmcr else np.random.uniform(lb[j], ub[j]) 
                                    for j in range(self.dim)])

            if np.random.rand() < par:
                gradient_approximation = (ub - lb) / self.dim * (-1 + 2 * np.random.rand(self.dim))
                new_harmony += self.learning_rate * gradient_approximation
                new_harmony = np.clip(new_harmony, lb, ub)

            if np.random.rand() < adaptive_quantum_prob:
                center = (np.mean(harmony_memory, axis=0) + best_harmony) / 2
                delta = np.abs(best_harmony - harmony_memory[np.random.randint(dynamic_memory_factor)])  # Change 7: Use dynamic memory factor
                dynamic_delta_factor = np.sqrt(2) * (1 + progress)
                new_harmony = center + np.random.uniform(-1, 1, self.dim) * delta / dynamic_delta_factor
                new_harmony = np.clip(new_harmony, lb, ub)

            score = func(new_harmony)
            self.evaluations += 1

            if score < best_score:
                best_score = score
                best_harmony = new_harmony
                self.learning_rate = max(0.05, self.learning_rate * 0.95)  # Change 8: Slightly increased decay rate of learning rate

            worst_idx = np.argmax(harmony_scores)
            if score < harmony_scores[worst_idx]:
                harmony_memory[worst_idx], harmony_scores[worst_idx] = new_harmony, score

        return best_harmony, best_score