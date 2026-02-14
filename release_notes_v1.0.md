# SOPilot Demo Outputs v1.0

## 📦 Contents

This release contains publication-quality figures and summaries proving SOPilot's neural scoring pipeline.

### 12 Figures (PNG, 200 DPI)
1. **Neural Pipeline Visualization** (6 figures)
   - Soft-DTW alignment with γ sweep
   - Alignment comparison (Cosine vs DTW vs Soft-DTW vs OT)
   - MC Dropout + Conformal prediction intervals
   - DILATE decomposition (shape vs temporal)
   - Explainability (Integrated Gradients + sensitivity)
   - Architecture diagram

2. **Ablation Study** (5 figures)
   - Alignment ablation: **Soft-DTW 43000× discrimination** vs Cosine 5.9×
   - Gamma sensitivity analysis
   - DILATE decomposition breakdown
   - Scoring head vs heuristic formula
   - Uncertainty coverage: **Conformal 92%** vs MC Dropout 74.5%

3. **End-to-End Pipeline** (1 figure)
   - 10-panel comprehensive architecture visualization

4. **Training Convergence** (1 figure)
   - **Proof: 1.7 → 81.5 (+79.9 points, 100% success rate)**
   - 8-panel convergence analysis

### 2 JSON Summaries
- `ablation_summary.json`: Quantitative ablation results
- `training_summary.json`: Training convergence metrics

## 🎯 Key Results

- **Training works**: Heuristic 1.7±4.2 → Neural 81.5±2.0 (+79.9, 100%)
- **Soft-DTW superior**: 43000× discrimination ratio (perfect vs noisy)
- **Conformal reliable**: 92% coverage (vs 95% target, distribution-free)

## 🚀 Reproduction

See [README.md](https://github.com/maruyamakoju/sopilot#readme) for one-command reproduction.

## 📚 Documentation

- [README.md](https://github.com/maruyamakoju/sopilot#readme): Full system overview
- [ACCOMPLISHMENTS.md](https://github.com/maruyamakoju/sopilot/blob/master/ACCOMPLISHMENTS.md): Development summary
