# Scratchpad: Attribution-Aware Direct Decoder

## Core Idea

Train a direct decoder (single forward pass) that learns to map from a pretrained DDPM's encoder feature space to generated images. Enable feature-space manipulation via hooks to control which training samples influence generation. Ultimate goal: data attribution and sample inclusion/exclusion for privacy/interpretability.

## Current Implementation (v1)

### Architecture
```
Random Noise [B, 1, 28, 28]
    ↓
[Frozen DDPM Encoder] (t=None, timestep-agnostic)
    ↓
Bottleneck Features [B, 128, 7, 7] + Skip Connections
    ↓
[Hook Point] ← Feature manipulation happens here
    ↓
[Trainable Direct Decoder] (no TimeMLP, mirrors DDPM decoder)
    ↓
Clean Image [B, 1, 28, 28]
```

### Key Design Choices

1. **Why train on DDPM outputs instead of real MNIST?**
   - Learn DDPM's generative distribution (richer than training set)
   - Enables novel sample generation (not just memorization)
   - Attribution must map to DDPM's feature→image relationship
   - User's goal: Navigate DDPM's learned manifold, not recreate MNIST

2. **Why t=None for encoder?**
   - Feature space must be consistent for manipulation
   - Can't use varying timesteps during inference (unpredictable)
   - Removes timestep as confounding variable
   - Cleaner attribution: features encode "what to generate" not "denoising state"

3. **Why keep skip connections?**
   - Quality vs. purity tradeoff
   - Without skips: quality degradation significant
   - With skips: hook at single point (bottleneck) still tractable
   - Future: multi-scale hooks for finer control

4. **Why warm start decoder?**
   - Pretrained decoder already knows feature→image mapping (with timesteps)
   - Significantly faster convergence
   - Just needs to learn timestep-agnostic version

5. **Why 60k samples?**
   - Match MNIST size for distribution coverage
   - Ensures decoder sees full diversity of DDPM outputs
   - Computational trade-off: pre-generate offline vs. on-the-fly

### Training Strategy

- **Loss**: MSE(direct_output, ddpm_sampled_output)
- **Data**: 60k (noise, DDPM_output) pairs pre-generated
- **Split**: 90% train, 10% validation
- **Optimizer**: AdamW, LR=0.0001
- **Scheduler**: OneCycleLR (25% warmup, cosine)
- **EMA**: Applied to decoder for stable sampling

### Hook Mechanism

Single manipulation point at bottleneck features [B, 128, 7, 7]:

```python
def hook_fn(features):
    # Manipulate features here
    return modified_features

output = model(noise, hook_fn=hook_fn)
```

**Current hook types:**
- Include: Push toward target sample features
- Exclude: Push away from target sample features
- Class-conditional: Push toward class-average features
- Amplify/dampen: Scale feature magnitudes

## Attribution Workflow

1. **Extract reference features** for all MNIST training samples
   - Run encoder on entire training set
   - Store features [60k, 128, 7, 7]
   - ~350MB on disk

2. **Generate with control**
   ```python
   # Include samples 0-9
   target = ref_features[0:10].mean(dim=0)
   hooked_features = features + strength * (target - features)
   ```

3. **Measure attribution**
   ```python
   # Cosine similarity in feature space
   similarity = F.cosine_similarity(gen_features, ref_features)
   # High similarity → strong attribution to that sample
   ```

4. **Iterate**
   - Identify contributing samples
   - Create exclude hooks to remove their influence
   - Re-generate and measure again

## Open Questions & Future Directions

### Immediate Experiments (Post-Training)

1. **Optimal hook strength**
   - What's the sweet spot? 0.3? 0.5? 0.7?
   - Does it vary by sample/class?
   - Can we learn optimal strength automatically?

2. **Multi-sample hooks**
   - How to combine multiple include/exclude targets?
   - Weighted averaging? Sequential application?
   - Does order matter?

3. **Attribution quantification**
   - Is cosine similarity the right metric?
   - Try: L2 distance, dot product, learned metric?
   - How to threshold "significant" attribution?

4. **Generalization**
   - Do hooks transfer across different noise seeds?
   - Can we find "universal" include/exclude directions?

### Architecture Improvements

1. **Multi-scale hooks**
   - Hook at each decoder layer, not just bottleneck
   - Coarse control (early layers) vs. fine details (late layers)
   - More complex but more powerful

2. **Learned hook functions**
   - Instead of manual manipulation, train hook networks
   - Input: target indices → Output: feature transformation
   - Meta-learning approach

3. **Feature space structure**
   - PCA on bottleneck features - what are principal components?
   - Cluster analysis - do classes separate?
   - Interpolation smoothness - is feature space well-behaved?

4. **Conditional encoder**
   - Current: encoder sees only noise
   - Alternative: condition encoder on target attributes
   - Closer to conditional GAN/VAE

### Loss Function Enhancements

1. **Perceptual loss (LPIPS)**
   - MSE penalizes pixel-level differences
   - Perceptual loss focuses on semantic similarity
   - Likely improves visual quality

2. **GAN discriminator (DMD2 approach)**
   - Discriminator: real DDPM outputs vs. direct decoder outputs
   - Forces distribution matching, not just pixel matching
   - More complex training but better quality

3. **Consistency loss**
   - Multiple noise seeds → same semantic content
   - Encourages stable feature space
   - Reduce sensitivity to random initialization

4. **Attribution-aware loss**
   - Explicitly penalize unwanted attributions during training
   - Train decoder to ignore certain samples
   - Requires pre-defined exclusion set

### Theoretical Questions

1. **Feature space linearity**
   - Are include/exclude directions linear?
   - Can we do feature arithmetic like GANs (smile - male + female)?
   - Test: interpolation smoothness

2. **Information bottleneck**
   - How much information loss at bottleneck [128, 7, 7]?
   - Compare: decode with vs. without skip connections
   - Quantify: reconstruction error, FID score

3. **Timestep dependence**
   - Does t=None lose important information?
   - Compare: train decoders with t=None, t=500, t=random
   - Hypothesis: t=None forces timestep-invariant features

4. **Attribution transitivity**
   - If A influences B, and B influences C, does A influence C?
   - Graph analysis of attribution network
   - Identify "influential" vs "derivative" samples

### Scalability & Efficiency

1. **Faster data generation**
   - Current: 1000-step DDPM sampling (slow)
   - Try: DDIM with 50 steps, distilled models
   - Or: Train decoder iteratively (bootstrap)

2. **Compression**
   - Do we need 60k samples or is 10k enough?
   - Active learning: select most informative pairs
   - Curriculum learning: easy → hard samples

3. **Online training**
   - Instead of pre-generating, sample on-the-fly
   - Cache DDPM outputs to avoid recomputation
   - Asynchronous data generation

### Privacy & Unlearning Applications

1. **Sample removal (machine unlearning)**
   - User requests: "forget sample X"
   - Apply strong exclude hook for sample X
   - Measure: can model still reconstruct X?

2. **Differential privacy**
   - How much information does decoder leak about training samples?
   - Membership inference attacks
   - Add noise to hooks for privacy guarantees

3. **Concept erasure**
   - Remove entire concepts (e.g., "all 3s")
   - Compare: exclude vs. retrain from scratch
   - Efficiency: O(1) hook vs. O(n) retraining

### Evaluation Metrics

**Quality Metrics:**
- FID (Fréchet Inception Distance) vs. real MNIST
- Inception Score
- Visual quality (human evaluation)

**Attribution Metrics:**
- Cosine similarity distribution
- Precision/recall of identified samples
- Exclusion effectiveness (generate without target samples)

**Efficiency Metrics:**
- Inference time: Direct decoder vs. DDPM (should be ~1000x faster)
- Memory usage
- Hook overhead

## Potential Issues & Mitigations

### Issue 1: Mode Collapse
**Symptom**: Decoder generates similar images regardless of noise
**Causes**:
- MSE loss too simple
- Dataset too small
- Training too long

**Solutions**:
- Add GAN discriminator
- Increase dataset diversity
- Monitor validation loss, early stopping
- Add noise injection during training

### Issue 2: Hook Instability
**Symptom**: Hooks have unpredictable effects
**Causes**:
- Feature space not well-structured
- Strength too high
- Interference between skip connections and hooks

**Solutions**:
- Normalize features before/after hooks
- Adaptive strength based on feature magnitudes
- Regularize feature space during training (e.g., L2 penalty)
- Remove skip connections (quality tradeoff)

### Issue 3: Poor Attribution
**Symptom**: Cosine similarity doesn't correlate with visual similarity
**Causes**:
- Feature space not aligned with semantic space
- Bottleneck too low-dimensional
- Skip connections bypass bottleneck

**Solutions**:
- Use multi-scale features (not just bottleneck)
- Increase bottleneck dimension
- Train with attribution-aware loss
- Use learned similarity metric instead of cosine

### Issue 4: Computational Cost
**Symptom**: Data generation or training too slow
**Causes**:
- 1000-step DDPM sampling
- Large batch sizes
- Device limitations (CPU vs. GPU)

**Solutions**:
- Use DDIM (50 steps instead of 1000)
- Reduce batch size, accumulate gradients
- Use mixed precision training (FP16)
- Distributed training across GPUs

## Alternative Approaches (Not Implemented)

### Approach A: VAE-Style Latent Space
Instead of using DDPM encoder, train a VAE on DDPM outputs:
- Encoder: Image → latent z
- Decoder: latent z → Image
- Loss: Reconstruction + KL divergence
- **Pro**: Clean latent space, easier manipulation
- **Con**: Doesn't leverage pretrained DDPM, more complex

### Approach B: Classifier-Guided Manipulation
Train classifier on encoder features:
- Classifier: features → sample identity
- Gradient-based manipulation toward/away from samples
- **Pro**: Differentiable, precise control
- **Con**: Requires labels, more complex optimization

### Approach C: Energy-Based Models
Model feature space as energy landscape:
- Low energy = likely features
- Manipulation = move along energy gradients
- **Pro**: Principled probabilistic framework
- **Con**: Training difficulty, computational cost

### Approach D: Neural ODE
Model feature transformation as continuous dynamics:
- ODE: dx/dt = f(x, t)
- Solve ODE to transform features
- **Pro**: Smooth, invertible transformations
- **Con**: Complex, slow inference

## Success Criteria (How to Know It's Working)

### Minimum Viable Product (MVP)
✅ Direct decoder generates recognizable MNIST digits
✅ Quality comparable to DDPM (visual inspection)
✅ Inference ~1000x faster than DDPM
✅ Hooks visibly affect generated images
✅ Attribution identifies top-5 similar samples correctly

### Good Performance
- FID < 10 (comparable to DDPM)
- Include hook generates images resembling target samples
- Exclude hook successfully removes visual features
- Feature space interpolation smooth (no artifacts)

### Excellent Performance
- FID < 5 (better than DDPM due to distillation)
- Attribution precision > 90% (top-10 samples)
- Unlearning: excluded samples not reconstructible
- Generalizes to novel compositions (e.g., include[0,1] + exclude[2,3])

## Experiment Tracking

### Ablation Studies to Run

1. **Skip connections**: With vs. without
2. **Timestep**: t=None vs. t=500 vs. t=random
3. **Training data**: 10k vs. 60k samples
4. **Loss function**: MSE vs. LPIPS vs. GAN
5. **Hook strength**: 0.1, 0.3, 0.5, 0.7, 0.9
6. **Decoder initialization**: Random vs. pretrained

### Hyperparameter Sweep

- Learning rate: [1e-5, 5e-5, 1e-4, 5e-4]
- Batch size: [32, 64, 128, 256]
- Epochs: [20, 50, 100, 200]
- EMA decay: [0.99, 0.995, 0.999]

## Long-Term Vision

**Phase 1 (Current)**: Basic direct decoder with single-point hooks
- Proof of concept
- Establish baseline quality
- Validate attribution approach

**Phase 2 (Next)**: Enhanced quality and attribution
- LPIPS/GAN loss for better visuals
- Multi-scale hooks for finer control
- Comprehensive attribution metrics

**Phase 3 (Future)**: Advanced applications
- Privacy-preserving generation (differential privacy)
- Machine unlearning at scale
- Interpretability tools for diffusion models

**Phase 4 (Research)**: Theoretical understanding
- Feature space geometry
- Attribution dynamics
- Provable guarantees for unlearning

## Related Work to Explore

- **Consistency Models** (Song et al.): Single-step generation
- **Progressive Distillation** (Salimans & Ho): Multi-step to few-step
- **DMD2** (Yin et al.): Distribution matching distillation
- **Machine Unlearning**: SISA, Fisher forgetting, influence functions
- **GAN Inversion**: Finding latent codes for real images
- **Concept Bottleneck Models**: Interpretable intermediate representations

## Open Research Questions

1. **Theoretical**: Is there a optimal feature space dimensionality for attribution?
2. **Empirical**: How does hook strength relate to attribution accuracy?
3. **Practical**: Can we automate hook design (meta-learning)?
4. **Ethical**: What privacy guarantees can feature manipulation provide?
5. **Scalability**: Does this approach work for high-res images (e.g., 256x256)?

## Notes from Planning Discussion

**User's motivation**:
- Attribution and avoidance of contributing images
- Include/exclude specific samples from generation
- Eventually navigate feature space to control outputs

**Why not train on real MNIST?**
- Would limit to known samples
- Goal is to reproduce DDPM's generative distribution (novel samples)
- Need to understand what DDPM learned, not just MNIST

**Why hooks in feature space?**
- Direct control over generation process
- Real-time manipulation (no retraining)
- Interpretable: push toward/away from specific samples

**Future hooks across layers:**
- Start simple (bottleneck only)
- Later: multi-scale (each decoder layer)
- Trade-off: simplicity vs. granularity

## Implementation Status

✅ Core architecture implemented
✅ Training pipeline complete
✅ Hook mechanism functional
✅ Attribution utilities created
✅ Comprehensive documentation written
⏳ Awaiting execution (data generation + training)
⏳ Experimental validation pending

## Next Session TODO

1. Run small test (100 samples) to validate code
2. If successful, start 60k data generation overnight
3. Train decoder for 50 epochs
4. Analyze results:
   - Sample quality
   - Hook effectiveness
   - Attribution accuracy
5. Plan Phase 2 improvements based on results

---

**Last updated**: 2025-11-18
**Status**: Implementation complete, awaiting experimental validation
**Primary goal**: Attribution-aware generation with sample inclusion/exclusion control
