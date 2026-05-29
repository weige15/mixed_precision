# Slide 1: Bayesian Bits: Problem and Research Gap
**Visual: comparison-3**

[Column 1: Deployment bottleneck]
- Neural-network inference cost is driven by arithmetic and data movement.
- Quantization lowers operand bit-width; pruning removes computation.
- Real deployments need accuracy-efficiency trade-offs, not only smaller files.

[Column 2: Past direction]
- Fixed-bit quantization learns scales or clipping for one precision.
- Mixed-precision work searches layer/tensor bit-widths using Hessians, RL, NAS, or differentiable objectives.
- Pruning is usually optimized as a separate compression problem.

[Column 3: What was missing]
- Search space grows exponentially with the number of tensors.
- Learned bit-widths may not map to hardware-friendly power-of-two precisions.
- Quantization and pruning lacked one gradient-based formulation.

---

# Slide 2: Why Existing Mixed Precision Was Not Enough
**Visual: cards-4**

[Card 1: Fixed-bit QAT]
- PACT, LSQ, and TQT learn clipping or step sizes.
- Strong at one selected bit-width.
- They do not decide where each tensor should spend bits.

[Card 2: Sensitivity search]
- HAWQ-style methods use Hessian signals to rank layers.
- They expose layer sensitivity.
- They still need a discrete assignment step across many configurations.

[Card 3: NAS / RL search]
- Architecture-search and reinforcement-learning methods can optimize bit policies.
- Some can incorporate hardware feedback.
- Search cost and policy complexity increase with model size.

[Card 4: Differentiable bits]
- Differentiable Quantization learns continuous bit-widths.
- The learned widths can require rounding for real hardware.
- Rounding can erase part of the promised efficiency gain.

---

# Slide 3: Core Idea: Gated Residual Quantization
**Visual: process-4-phase**

[Column 1: Start low]
- Quantize each tensor first at 2-bit.
- This gives the lowest nonzero grid used by the method.

[Column 2: Add residuals]
- Quantize the remaining error to move from 2 to 4, then 8, 16, and 32 bits.
- Each residual doubles the effective precision.

[Column 3: Learn gates]
- Binary stochastic gates decide whether each residual is included.
- If a lower gate is off, all higher-precision residuals are off too.

[Column 4: Unify pruning]
- Add a 0-bit gate before the 2-bit value.
- Switching it off makes pruning a special case of quantization.

---

# Slide 4: Observations, Results, and Takeaway
**Visual: table**

| Observation | Evidence / implication |
|---|---|
| Low-bit priors create useful pressure. | Gates are regularized through a Bayesian variational objective; the prior can be scaled by BOP cost. |
| Joint pruning + quantization is better than either alone. | ImageNet ResNet18 ablations show the combined Bayesian Bits curve dominates pruning-only and quantization-only variants. |
| The learned policy recovers familiar heuristics automatically. | Aggressive settings push most tensors to 2-bit, while often preserving first and last layers at higher precision. |
| It works beyond end-to-end fine-tuning. | Post-training experiments learn gates, or gates plus scales, on a pretrained ResNet18 without updating weights. |
| Main limitation: training overhead. | ResNet18 experiments used 30 epochs of Bayesian Bits plus 10 fixed-gate fine-tuning epochs, about 70 hours on one Tesla V100. |

