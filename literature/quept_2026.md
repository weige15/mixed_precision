# QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching

- Authors: Ke Xu, Yixin Wang, Zhongcheng Li, Hao Cui, Jinshui Hu, Xingyi Zhang
- Venue/year: AAAI 2026
- Source: https://arxiv.org/abs/2602.12609
- PDF: https://ojs.aaai.org/index.php/AAAI/article/download/39945/43906
- Code: https://github.com/xuke225/QuEPT
- Topic: Elastic precision Transformers with one-shot calibration

## Key Idea

QuEPT supports real-time switching among predefined bitwidths without repeated
optimization. It uses Multi-Bit Token Merging and Multi-Bit Cascaded LoRA to
manage cross-bit interactions and reconstruct block-wise multi-bit errors from a
small calibration slice.

## Why It Is Like MoBiQuant

Both papers target elastic precision. MoBiQuant reconstructs higher precision
from residual bit slices selected by a token-aware router; QuEPT uses one-shot
calibration and cascaded low-rank adapters so one Transformer can switch among
bitwidths.

## Fit To Soft Pruning

QuEPT is useful for the soft-to-hard transition. During calibration, multiple
bitwidth paths are jointly optimized; at deployment, the model can switch among
resource levels rather than making a single irreversible precision choice.

