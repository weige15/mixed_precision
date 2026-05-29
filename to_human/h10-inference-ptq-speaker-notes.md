# H10 Inference PTQ Progress Report Speaker Notes

## Slide 1

各位好，今天報告的是 H10：Hardware-Calibrated Precision Assignment，簡稱 HCPA。這個題目要解決的問題很直接：在已經訓練好的大型語言模型上，我們怎麼知道哪些部分可以安全地換成低精度，並且真的在目標 inference backend 和硬體上得到速度或吞吐量的好處。

目前這版進度的重點，是我們已經完成第一個嚴格門檻下的正向部署結果。以 matched bf16 為基準，GPTQ-Marlin 在三種 workload 上 prompt-NLL degradation 只有約 0.774%，低於 1% 的 strict gate；同時 latency 降低約 60 到 63%，output token throughput 提升約 153 到 168%。所以這裡的核心訊息是：HCPA 不只是提出方法，而是已經有一個 backend-real、quality-preserving 的 inference policy 可以支撐。

## Slide 2

這一頁先定義 HCPA 的研究目標。HCPA 可以看成一個 backend-aware 的 precision assignment loop：輸入是一個固定的 pretrained LLM，不做訓練階段的 policy tuning；接著我們估計不同 module 或 group 對低精度的敏感度，再只列出目標 backend 真的支援的 precision actions。

這裡很重要的是「只討論跑得起來的 action」。很多低精度方法在論文上看起來有效，但如果 target stack 沒有 kernel、memory layout 或 runtime 支援，最後就不能變成部署策略。因此 HCPA 會把 latency、memory、throughput、quality 和 failure evidence 都放進 action table，再交給 solver 找出品質可接受、部署指標有改善的 policy。

## Slide 3

這一頁說明 HCPA 跟 HAQ 的關係。HAQ 對我們最有用的不是特定的 RL controller，而是它把 precision selection 定義成一個 hardware-aware assignment problem：不同 layer 可以有不同 bitwidth，而且選擇要受硬體 cost 和 accuracy constraint 共同限制。

HCPA 把這個思想翻譯到 LLM inference。原本的 layer-wise bitwidth search，換成 module 或 group 的 precision-action assignment；原本的 latency 或 energy feedback，換成實測的 backend latency、memory、throughput 和 feasibility；accuracy constraint 則換成 prompt-NLL gate，再加上 task-quality sanity check。也就是說，我們保留 HAQ 的 assignment framing，但把搜尋空間換成現在 LLM backend 真正能執行的 action rows。

## Slide 4

這一頁把 HCPA 的 heuristic sources 放在一起。不同量化工作其實提供了不同層次的線索：HAQ 和 HAWQ 給的是 hardware-aware 和 sensitivity-aware assignment 的動機；GPTQ 和 AWQ 提醒我們，權重重建誤差和 activation-aware protection 會影響哪些模組適合低精度；SmoothQuant 和 LLM.int8() 則指出 activation outliers 是量化難度的重要訊號。

但 HCPA 的角色不是直接相信某一篇方法的 heuristic，而是把這些候選訊號轉成可測量、可求解的 action table。像 SpQR、OWQ 和 SqueezeLLM 會啟發 selective rescue 或 mixed-action design space；vLLM、Marlin 和 TorchAO 則提醒我們 deployment value 取決於 kernel 和 backend support。最後仍然要回到目標模型、目標 backend 和目標硬體上驗證。

## Slide 5

方法上，HCPA 最核心的 artifact 是 table/solver interface。每一列 candidate row 都必須回答四件事：它改了模型的哪個部分、backend 是否支援、品質風險多高，以及在目標硬體上實測成本是多少。

流程可以分成五步。第一步估計 module 或 group 對低精度的敏感度；第二步只列舉 backend-supported actions；第三步實測 latency、memory、throughput、quality 和 failure；第四步用 solver 在品質限制下選出有 deployment improvement 的 assignment；第五步把 policy freeze，並清楚寫出 claim boundary。這個 contract 可以避免三種過度宣稱：只改善 decode 但整體不成立、品質其實沒過 gate，或 action 根本不是 backend-supported。

## Slide 6

這一頁是 strict solver 的狀態。這次輸入共有 18 個 candidate rows，包含 AWQ 和 GPTQ 的 artifact summaries。經過 feasibility 和 1% prompt-NLL gate 後，有 15 列是 quality-passing feasible；再加上 deployment metric improvement 和 Pareto filtering，最後留下 7 個 accepted frontier rows。

被拒絕的原因也很有資訊量。AWQ-Marlin 速度快，但是 strict quality gate 沒過；bf16 是 reference，不會被選成改善項；部分 fp16 rows 雖然風險小，但沒有實質改善 deployment metric。真正留下來、也最有意義的是 GPTQ-Marlin：它同時通過 strict quality gate，並在 latency 和 output throughput 上有大幅改善。因此這一頁要傳達的是，solver 已經能把「快但品質不夠」和「安全但沒有部署收益」分開。

## Slide 7

主結果在這一頁。GPTQ-Marlin 是目前第一個 strict positive HCPA deployment result。它在三種 matched Instruct workload 上都通過品質門檻，並且有部署級的速度收益。

具體來看，decode-heavy workload latency 降低 62.7%，output tok/s 提升 168.3%；mixed workload latency 降低 62.4%，throughput 提升 166.3%；prefill-heavy workload latency 也降低 60.5%，throughput 提升 153.1%。品質代價是 prompt-NLL delta 約 0.773771%，在鎖定的 1% strict gate 以內。需要注意的是，這條 vLLM measurement path 的 memory deltas 是小幅正值，所以目前 claim 應該是 latency/throughput frontier，而不是 memory saving。

## Slide 8

這一頁聚焦在品質門檻，因為它決定哪些 speedup 可以被當成最終結果。fp16 default 的 prompt-NLL delta 很小，約 0.009738%，因此可以被視為低風險 baseline option，但它的速度改善有限。GPTQ-Marlin 的 delta 約 0.773771%，雖然比 fp16 高，但仍在 1% strict gate 內，而且帶來最大的 latency 和 throughput 改善，所以被 solver 選中。

AWQ-Marlin 的情況則不同。它同樣是 backend-feasible，而且速度上有吸引力，但 prompt-NLL delta 到 2.851377%，超過 strict gate。因此它不能作為 H10 的最終 strict result，只能被描述成 relaxed 3% threshold 下的 sensitivity path。這一頁的 takeaway 是：HCPA 支持 GPTQ-Marlin 作為 strict quality-preserving policy，而不是把所有 fast artifact 都混在一起宣稱成功。

## Slide 9

除了 prompt-NLL，我們也做了一個小型 task-quality screen。這不是完整 benchmark suite，但可以檢查量化後是否出現相對於 bf16 或 fp16 的新 failure mode。

結果顯示 bf16、fp16 和 GPTQ-Marlin 的 raw exact match 都是 0.6，adjusted accuracy 都是 0.8，而且 pass/fail pattern 相同。它們都通過 sentiment、sequence continuation、yes/no，以及經過 numeric rescoring 後的 arithmetic。唯一共同失敗的是 decimal comparison prompt：所有 policy 都回答 0.11 而不是 0.9。因為這個失敗不是 GPTQ-Marlin 特有，而是所有 policy 共享，所以目前可以說 GPTQ-Marlin 在這個 sanity check 上沒有引入新的 behavior regression。

## Slide 10

這一頁說明 layer/group path，也就是 HCPA 下一步要真正走向 mixed assignment 的地方。Whole-artifact GPTQ 已經給我們一個強的 first instance，但完整 HCPA 目標是對 module 或 group 做 precision choices，因此需要 backend-supported 的 module/group actions，而且這些 actions 必須在實測成本上真的有優勢。

目前 pipeline 已經能吃 layer/group rows。Transformers 加 TorchAO FQN path 已經實作，bf16/fp16 baselines 和 late MLP group candidates 也有資料；late gate/up INT8 weight-only 的品質可以，但 latency 和 memory 反而比 bf16 差，所以沒有被選；late MLP INT8 weight-only 目前品質資料還不完整。最後 H10 layer/group solver 在 14 個 input rows 中選出 0 個，原因不是 solver 不能運作，而是當前 backend/action space 還沒有找到 deployment metric 會改善的候選。

## Slide 11

這一頁是 claim boundary，目的是把已經被證據支持的結論和還不能宣稱的部分分清楚。現在可以支持的結論包括：H10 的 action-table 和 solver pipeline 已經可以執行；matched GPTQ-Marlin 通過 strict 1% prompt-NLL gate；latency 約下降 60 到 63%，output throughput 約提升 153 到 168%；小型 task-quality screen 沒有看到相對於 bf16/fp16 的 regression。

但也有幾個邊界要明講。第一，這次 final vLLM path 不能宣稱 memory saving。第二，結果目前集中在一個 Instruct artifact family，還不能外推到所有模型或所有 artifact。第三，task-quality screen 仍然只是 sanity check，還不是完整 downstream benchmark suite。第四，layer/group mixed-precision backend result 目前還沒有 positive selection。把邊界講清楚，反而能讓 HCPA 的貢獻更可信。

## Slide 12

最後是 future endeavor。下一步的目標，是把 HCPA 從一個成功的 backend artifact，推進成可重複的 precision-assignment method。第一件事是 replication：用第二個模型或第二個 artifact，在同一個 strict gate 下重跑，確認結果不是單一案例。

第二是建立 module/group risk model，在 solver 之前先估計每個 group 的 sensitivity。第三是擴充 action grid，列出 backend 真的支援的 precisions 和 formats。第四是系統化 cost table，持續量測 memory、latency、throughput、quality 和 failures。最後，solver 要能 freeze 出一個 mixed assignment，並清楚呈現 trade-off。整體研究目標仍然是：給定 pretrained LLM 和 target hardware，選擇每個 module 或 group 的 precision，在保留品質的同時改善 memory 或 latency。
