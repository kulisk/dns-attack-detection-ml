#!/usr/bin/env python3
"""Add missing methods to ensemble detector."""

content = open("src/models/supervised/ensemble_neural_detector.py", "r", encoding="utf-8").read()

# Check if the method already exists
if "def generate_unlabeled_data" not in content:
    # Add _UnlabeledDataGenerator class at the top 
    if "class _UnlabeledDataGenerator:" not in content:
        # Find the location after "logger = get_logger(__name__)" line
        insert_pos = content.find("logger = get_logger(__name__)")
        insert_pos = content.find("\n", insert_pos) + 1
        
        generator_class = '''

# ──────────────────────────── Unlabeled Data Generation ─────────────────────────

class _UnlabeledDataGenerator:
    """Generate synthetic unlabeled benign DNS data for consistency regularization."""
    
    @staticmethod
    def generate_benign_unlabeled(
        n_samples: int,
        input_dim: int,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Generate synthetic benign (unlabeled) DNS traffic features."""
        rng = np.random.default_rng(random_state)
        X = np.zeros((n_samples, input_dim))
        
        for i in range(min(input_dim, 6)):
            if i == 0:  # query_length
                X[:, i] = rng.normal(30, 10, n_samples)
            elif i == 1:  # entropy
                X[:, i] = rng.normal(3.5, 0.5, n_samples)
            elif i == 2:  # num_subdomains
                X[:, i] = rng.poisson(1, n_samples)
            elif i == 3:  # num_labels
                X[:, i] = rng.poisson(2, n_samples)
            elif i == 4:  # max_label_length
                X[:, i] = rng.normal(10, 3, n_samples)
            elif i == 5:  # avg_label_length
                X[:, i] = rng.normal(6, 2, n_samples)
        
        X[:, 6:] = rng.uniform(0.2, 0.8, (n_samples, input_dim - 6))
        X = np.clip(X, 0, 100)
        return X.astype(np.float32)

'''
        content = content[:insert_pos] + generator_class + content[insert_pos:]
    
    # Add generate_unlabeled_data method after get_params
    get_params_pos = content.find("    def get_params(self)")
    get_params_end = content.find("    def save(self", get_params_pos)
    
    method = '''    def generate_unlabeled_data(self, n_samples: int = 5000) -> np.ndarray:
        """Generate synthetic unlabeled benign DNS data.
        
        Useful for semi-supervised learning without real benign traffic data.
        
        Args:
            n_samples: Number of unlabeled samples to generate.
        
        Returns:
            Numpy array of shape (n_samples, input_dim) with synthetic benign data.
        """
        return _UnlabeledDataGenerator.generate_benign_unlabeled(
            n_samples=n_samples,
            input_dim=self.input_dim,
            random_state=42,
        )

'''
    
    content = content[:get_params_end] + method + content[get_params_end:]
    
    with open("src/models/supervised/ensemble_neural_detector.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✓ Added generate_unlabeled_data method and _UnlabeledDataGenerator class")
else:
    print("✓ Method already exists")
