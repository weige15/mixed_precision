# runtime_adaptive_mpq.pptx 報告口說稿

## Slide 1. Proposed Research 1: Runtime-Adaptive Mixed-Precision Quantization for LLMs

這一頁我想先說明整個研究的主軸。

過去的量化方法通常把 precision 當成一個模型部署前就決定好的靜態屬性，例如整個模型使用 4-bit，或是某些 layer 固定使用比較高的 bit-width。這樣做雖然可以降低記憶體與頻寬成本，但它有一個問題：不同 query、不同 layer、甚至不同 token，其實對 precision 的需求是不一樣的。

所以這個研究的核心想法是：precision 不應該只是靜態模型屬性，而應該被視為一種 runtime resource。也就是說，在推論時，我們根據 query difficulty、layer sensitivity，以及 token-level outlier behavior，動態決定哪些地方需要花比較多 bit，哪些地方可以省下來。

這裡我把三篇相關工作串成一條線。

第一個是 QAQ，它從 query 層級出發，根據 request 的難度預測 precision budget，以及可能需要的 memory movement。

第二個是 DP-LLM，它從 layer 層級出發，根據 relative error 判斷哪些 layer 在目前 decoding step 需要比較高的 precision。

第三個是 MoBiQuant，它進一步到 token 層級，使用 residual bit slices，讓 sensitive token 可以啟用更多 bit slice，而不敏感的 token 使用較低 precision。

我的 proposed research 是在這三者之上，再加上一個 hardware-friendly 的角度：如果某些 adaptive precision pattern 很常出現，我們就不應該每次都完全動態判斷，而是把它們整理成穩定、可重複使用的 precision highway。

換句話說，目標不是單純讓 routing 越動態越好，而是保留 adaptivity 的好處，同時讓常見路徑變得可預測、可最佳化。

## Slide 2. Mechanism

這一頁是在說明整個方法的基本機制。

我的想法是把 precision decision 拆成三個層級。

第一層是 query level。QAQ 負責判斷這個 request 大概需要多少 precision budget。例如簡單 query 可以用比較低的 bit-width，困難 query 或 outlier-heavy query 則需要比較高 precision。

第二層是 layer level。DP-LLM 的角色是判斷目前 decoding step 中，哪些 layer 的 quantization risk 比較高。這裡可以用 relative error 來衡量，也就是低 precision 和高 precision weight 造成的輸出差異。

第三層是 token level。MoBiQuant 負責執行更細粒度的 precision allocation。透過 residual bit slices，模型可以只在 sensitive token 上啟用額外 slice，避免每個 token 都使用高 precision。

所以整體流程可以理解成：

先由 query 決定大方向的 budget，再由 layer 判斷哪些地方風險高，最後由 token-level bit slices 做精細化執行。

我的研究重點是：這些決策如果每次都完全動態做，會造成 router overhead、irregular memory access，以及 kernel fusion 困難。因此，我們希望從歷史或 calibration trace 中找出常見 pattern，把它們轉成可最佳化的 precision highway。

## Slide 3. Step 1: Calibrate

接下來是 proposed research 的第一步：Calibrate。

這裡的 calibrate 不是要直接訓練出最終方法，而是要收集 runtime signals，建立後續 mining 和 compile 所需要的 trace data。

我目前定義四種 calibration trace。

第一個是 query difficulty。這代表一個 prompt 或 request 的複雜度，例如 prompt 長度、entropy、task type，或是 router 給出的 difficulty score。它的目的是估計這個 query 大概需要多少 precision budget。

第二個是 layer relative error。這是 DP-LLM 裡面的概念，用來估計某一層在低 bit 和高 bit weight 之間造成的輸出差異。如果某一層的 relative error 很大，就代表這一層在目前輸入下比較不適合用低 precision。

第三個是 token outlier traces。這來自 MoBiQuant 的觀察：不同 bit-width 下，造成 quantization error 的 token 不一定相同，也就是 outlier migration。因此我們要記錄哪些 token 在不同 bit-width 下反覆造成比較大的 quantization error。

第四個是 system budgets。這代表目前系統還有多少 latency、memory、bandwidth，或 energy slack 可以拿來使用高 precision。因為即使模型想用高 precision，如果硬體資源不允許，還是需要降低 bit-width。

所以 calibrate 的輸出可以想成一張 trace table。它記錄每個 query、layer、token 在不同 system budget 下，對 precision 的需求與風險。

這一步的重點是收集證據，而不是做最終決策。

## Slide 4. Step 2: Pattern Discovery & Compile

第二步是 Pattern Discovery 和 Compile。這裡其實可以再拆成兩個概念：mine 和 compile。

Mine 或 pattern discovery 的目的是分析前面 calibration 得到的 trace，找出穩定、常見，而且低風險的 precision pattern。

例如，我們可能會發現，hard math query 常常在後段 MLP layers 需要比較高的 precision；或者 long-context prompt 常常在 attention blocks 啟用額外 bit slices；也可能發現某些 layer group 經常一起切換 precision。

這些 pattern 如果出現頻率很高，而且在不同 calibration sample 上都很穩定，就代表它們有機會被整理成 precision highway。

接著是 compile。Compile 的意思是把這些被發現的 pattern 轉成真正可以在硬體上有效執行的 fast paths。

具體來說，第一個方向是 kernel fusion。假設某些 low precision 和 high precision operation 經常一起執行，我們可以把它們合併成較少的 GPU kernels，減少 kernel launch overhead 和 memory traffic。

第二個方向是 reduced router overhead。如果目前 query 很明確地符合某個 high-confidence pattern，就不需要每一層、每個 token 都跑完整 router，而是可以直接走已經編譯好的 precision highway。

第三個方向是 memory pre-planning。因為我們已經知道某些 bit-planes 或 residual slices 很可能會被用到，所以可以事先規劃 memory layout 或 prefetch，減少 runtime 才臨時搬資料的成本。

所以簡單來說，Mine 是找出常見 pattern；Compile 是把這些 pattern 變成硬體友善的執行路徑。

這也是我和 QAQ、DP-LLM、MoBiQuant 不同的地方。這三篇主要是在說明如何做 adaptive precision，而我的方法進一步問：這些 adaptive decisions 能不能被整理成可預測、可最佳化的路徑？

## Slide 5. Step 3: Routing

最後一步是 Routing，也就是實際 inference 時要做的事情。

在推論期間，系統會先判斷目前 query 或 token 是否符合已經 mining 和 compile 出來的 precision highway。

如果目前 query 和某個已知 pattern 很接近，而且 confidence 夠高，並且當前 system budget 也允許，那 routing module 就直接把它送到對應的 compiled precision highway。這樣可以減少細粒度 router 的呼叫次數，也可以讓 memory access 和 kernel execution 更穩定。

如果目前 query 不符合任何高信心 pattern，或是系統 budget 不適合使用該 highway，那就回到比較完整的 dynamic selector，也就是使用 QAQ、DP-LLM、MoBiQuant 類似的動態 precision decision。

所以 compile 和 route 之間有關係，但不是同一件事。

Compile 是 before inference 的工作：準備 fast paths。

Route 是 during inference 的工作：根據目前輸入和系統狀態，選擇要走 fast path，還是要 fallback 到完整 dynamic routing。

整個流程可以總結成四步。

第一，Calibrate：收集 runtime signals，了解什麼時候額外 precision 是值得的。

第二，Mine：找出 high support、low variance、low quality risk 的 precision patterns。

第三，Compile：把這些 pattern 變成 fused kernels、memory prefetch plan，或其他硬體友善的 execution path。

第四，Route：推論時優先使用 stable highway，只有在 input 離開這些 highway 時，才啟用完整 runtime router。

最後，我希望這個方法能達到的目標是：保留 dynamic mixed-precision 的 accuracy benefit，同時降低 router overhead、減少 irregular memory access，並讓 adaptive precision 更容易真的轉換成 latency 和 memory improvement。

## One-Sentence Summary

這個研究可以用一句話總結：

我提出 Precision Highway，一個 runtime-adaptive mixed-precision framework。它先 profiling query-layer-token precision traces，接著 mining 穩定且常見的 precision patterns，再把這些 patterns compile 成 hardware-friendly fast paths，最後在推論時只有低信心或不符合 pattern 的情況才 fallback 到完整 dynamic routing。
