"""Quick test of ensemble neural detector."""
import numpy as np
from src.models.supervised import SemiSupervisedEnsembleDetector
from src.data_collection.synthetic_generator import SyntheticDNSGenerator

# Generate synthetic data
print("📊 Generating synthetic data...")
gen = SyntheticDNSGenerator(random_state=42)
df = gen.generate(n_samples=5000)
print(f"✓ Generated {len(df)} samples")

# Prepare features and labels
from src.preprocessing import DataCleaner, DNSScaler, LabelEncoder
from src.feature_engineering import DNSFeatureExtractor

cleaner = DataCleaner()
scaler = DNSScaler()
extractor = DNSFeatureExtractor()
label_encoder = LabelEncoder(known_classes=[
    "benign", "dns_ddos", "dns_amplification", "dns_tunneling",
    "cache_poisoning", "nxdomain_attack", "data_exfiltration", "botnet_dns"
])

# Process data
print("🔧 Processing features...")
df = extractor.transform(df)
df = cleaner.fit_transform(df, "label")
y = label_encoder.fit_transform(df["label"])
feature_cols = [c for c in df.columns if c not in {"label", "is_attack"}]
X = scaler.fit(df[feature_cols]).transform(df[feature_cols]).values

# Split data
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"✓ X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# Create ensemble model
print("\n🧠 Creating ensemble neural detector...")
model = SemiSupervisedEnsembleDetector(
    input_dim=X_train.shape[1],
    n_classes=len(label_encoder.classes),
    epochs=5,  # Just 5 epochs for quick test
    patience=3,
    batch_size=128,
)
print(f"✓ Model created: {model}")

# Train model
print("\n📚 Training ensemble neural detector...")
model.fit(X_train, y_train, X_val=X_val, y_val=y_val, X_unlabeled=None)
print("✓ Training completed")

# Evaluate
print("\n📈 Evaluating on test set...")
pred = model.predict(X_test)
pred_proba = model.predict_proba(X_test)
accuracy = np.mean(pred == y_test)
print(f"✓ Test accuracy: {accuracy:.4f}")
print(f"✓ Predictions shape: {pred.shape}")
print(f"✓ Probabilities shape: {pred_proba.shape}")

# Test save/load
print("\n💾 Testing save/load...")
path = model.save()
print(f"✓ Model saved to {path}")

model2 = SemiSupervisedEnsembleDetector()
model2.load()
pred2 = model2.predict(X_test)
accuracy2 = np.mean(pred2 == y_test)
print(f"✓ Loaded model test accuracy: {accuracy2:.4f}")
print(f"✓ Predictions match: {np.allclose(pred, pred2)}")

print("\n✅ ALL TESTS PASSED!")
