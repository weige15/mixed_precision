# runtime_adaptive_mpq.pptx：Introduction & Motivation 四頁口說稿

## Page 1. Introduction & Motivation

這一頁我想先建立研究動機：為什麼我們需要 mixed-precision soft pruning。

LLM inference 的成本主要來自兩個地方。第一個是計算量很大，模型每次推論都需要大量矩陣運算。第二個更重要的是 memory traffic，尤其在實際硬體上，很多時候瓶頸不是 GPU 算不動，而是權重、activation、cache 資料搬動太慢。也就是說，模型推論常常是 memory-bound，而不是單純 compute-bound。

所以 quantization 會有效，是因為它同時降低 arithmetic cost 和 memory bandwidth。但如果我們把所有權重都壓到同一個低 bit-width，例如全模型 4-bit，就會遇到 accuracy 問題。某些 layer、channel、token，或某些 prompt，其實對 precision 很敏感；如果這些地方也被壓太低，品質就會掉。

因此這裡我用 soft pruning 的角度來看 mixed precision。Soft pruning 不是把 weight 永久刪掉，而是讓重要資訊仍然可以用比較高的 precision 表示，不重要的資訊則使用比較少的 effective bits。換句話說，我們不是問「哪些 weight 要不要存在」，而是問「哪些資訊值得用更多 bits 保存」。

這帶出三個核心問題。第一，對目前這個 prompt 來說，哪些 component 需要更多 bits？第二，硬體成本是否值得我們花這些 bits？第三，如果不同 weight component 有不同 bit-width，要怎麼有效地存進記憶體並執行？

所以這一頁的重點是：precision 不應該只是部署前固定好的 compression setting，而應該被視為 runtime 可以動態分配的 resource。

## Page 2. Current Quantization: Mostly Static Decisions

這一頁是在整理目前量化研究的主要方向，以及它們共同的限制。

第一類是 uniform PTQ 或 QAT。這類方法會選一個全域格式，例如 W8A8、W4A16，或 NF4。它的好處很明顯：儲存格式簡單、kernel 也容易實作，因為所有權重基本上都遵守同一種 bit-width 或資料格式。但缺點是，它沒有辦法把額外 bits 分配給比較 fragile 的 component。

第二類是 outlier-aware static quantization。像 LLM.int8 會把 outlier channel 特別處理；SmoothQuant 會把 activation 的難度轉移到 weight；AWQ 則用 calibration activation 來保護 salient weights。這些方法都比單純 uniform quantization 更細緻，但它們大多還是在 calibration 後得到一個固定 policy。

第三類是 offline mixed precision。像 HAWQ 用 Hessian 訊號估計 layer sensitivity，HAWQ-V3 進一步把硬體 constraint 放進 bit assignment，DNAS 和 HAQ 則用 search 的方式找 mixed-precision policy。這些方法已經承認不同 layer 不應該用同一個 precision，這點很重要。

但是它們共同的限制是：最後選出的 bit policy 大多仍然是 static 的。也就是說，部署後，不管今天的 prompt 是簡單還是困難，不管 runtime hardware budget 是否改變，precision allocation 通常都不會隨輸入動態調整。

所以這一頁要鋪陳的是：現有方法已經從 uniform precision 走向 mixed precision，但還沒有真正把 precision 當成 runtime adaptive resource。

## Page 3. Why Static Precision Is Not Enough

這一頁進一步說明，為什麼 static precision 對我們想做的 soft pruning 不夠。

第一個原因是 prompt-dependent need。不同 prompt 對 precision 的需求不一樣。簡單 prompt 可能用比較少 bit planes 就足夠；但是困難 prompt、數學推理 prompt，或是會產生 activation outlier 的輸入，就可能需要更多 precision。如果我們用 static profile，就必須為了保護 worst case 而讓很多 easy case 也使用較高 precision，這會浪費資源。

第二個原因是 component sensitivity。模型裡面不同 component 對量化誤差的容忍度不同，例如 attention、MLP、embedding、outlier channels 都可能有不同 sensitivity。而且這個 sensitivity 不一定是固定的，它可能會隨著 context、task，甚至 decoding step 改變。Uniform bit-width 在這裡會浪費 budget，因為它會把相同 precision 給到其實不敏感的 component。

第三個原因是 system constraint。這裡的重點是：不同 bit-width 的 weight 很難有效地放進硬體喜歡的 memory layout。比較精確的說法是，heterogeneous bit-widths 會 complicate aligned memory packing and contiguous loads。

硬體通常喜歡固定寬度、對齊、連續的 memory access，例如 word、lane、cache line 這類單位。如果有些 weight 是 2-bit，有些是 4-bit，有些是 8-bit，它們的位址就不容易乾淨地對應到固定寬度的 memory unit。這可能需要額外 metadata、bit unpacking、masking，或比較不連續的 memory load。最後這些 overhead 可能會吃掉低 bit 帶來的加速。

所以動態 mixed precision 的目標不是只做一個更聰明的 algorithm，而是要同時考慮三件事：component sensitivity、prompt difficulty、hardware cost。我的初步 formulation 可以寫成：

precision = f(component sensitivity, prompt difficulty, hardware cost)。

這也是為什麼我目前的方向會是先 disentangle important 和 less-important weight information，再設計一種 storage 或 embedding 方式，讓 runtime 可以根據需求 materialize 不同 effective precision。

## Page 4. Heuristics from HAQ

最後這一頁是在說明，HAQ 對這個研究方向提供了什麼啟發。

HAQ 本身不是我要直接採用的最終方法。它主要是針對 CNN 或 DNN inference，使用 reinforcement learning 來搜尋 layer-wise weight 和 activation bit-width。而且它搜尋出來的 policy 是 static 的，並不是 prompt-adaptive。

但是 HAQ 的核心思想很有價值：precision allocation 應該被視為一個 policy，而且這個 policy 不應該只根據 FLOPs 或 model size 這種 proxy metric，而是要把 hardware feedback 放進 loop 裡。也就是說，policy 要根據 latency、energy、model size 這些實際 deployment cost 來調整。

在這一頁我把 HAQ 的想法轉成我們的 soft-pruning heuristic。

第一步是 signals。HAQ 原本看的是 layer state 和 hardware feedback；在我們的問題中，signals 會包含 component sensitivity、prompt difficulty，以及 hardware cost。

第二步是 policy。HAQ 使用 RL actor 找 static bit policy；我們可以把它改成 dynamic router 或 solver，根據目前輸入和硬體狀態決定 precision。

第三步是 bits。HAQ 的輸出是每層的 bit-width；我們這裡想要的是保留高價值 component，壓縮低價值 component，讓 precision 變成更細粒度的 information budget。

第四步是 runtime。這是 soft pruning 特別需要補上的地方：aligned packing、contiguous loads，以及 kernel-aware execution。因為如果 storage layout 和 kernel 不支援，動態 precision 在 algorithm 上看起來有效，實際 latency 可能不會變好。

所以可以把 HAQ 當成一個 heuristic：policy 加上 hardware feedback，再加上 deployment constraint。我的研究想把這個 pattern 往前推一步，加入 prompt adaptivity 和 heterogeneous-bit storage，讓 mixed precision 不只是 offline search 的結果，而是 runtime 可以使用的動態資源。

