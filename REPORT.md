# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Lê Trung — 2A202600174  
**Ngày nộp**: 2026-05-07  
**Submission option**: B (HuggingFace Hub + GitHub)

---

## 1. Setup

- **Base model**: `unsloth/Qwen3-4B-unsloth-bnb-4bit` (Qwen3 4B, 4-bit NF4 quantized)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval, random seed = 42)
- **max_seq_length**: 1024 (hard cap cho T4; p95 của dataset nằm trong giới hạn này)
- **GPU**: Tesla T4, 16 GB VRAM (Google Colab Free)
- **Training cost**: ~$0.30 (tổng ~50.6 phút training @ $0.35/hr)
- **HF Hub link**: https://huggingface.co/NLT22/qwen3-4b-vi-lab21-r16
- **GitHub link**: https://github.com/NLT22/lab21_2A202600174

**Cấu hình chung cho cả 3 lần train:**

| Hyperparameter | Giá trị |
|---|---|
| Epochs | 10 |
| LR scheduler | cosine |
| Learning rate | 2e-4 |
| Warmup ratio | 0.10 |
| Effective batch size | 8 (batch=1 × grad_accum=8) |
| Optimizer | adamw_8bit (paged AdamW) |
| target_modules | `q_proj`, `v_proj` |
| Gradient checkpointing | True (unsloth mode) |
| Packing | False |

---

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | % of Total | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|-----------------|------------|------------|-----------|-----------|------------|
| **8**  | 16  | 2,949,120  | 0.073% | 16.63 min | 10.97 GB | 1.537 | **4.65** |
| **16** | 32  | 5,898,240  | 0.146% | 16.87 min | 10.22 GB | 1.678 | 5.36 |
| **64** | 128 | 23,592,960 | 0.583% | 17.14 min | 12.01 GB | 2.182 | 8.86 |
| Base   | —   | —          | —      | —         | —        | —     | — |

> **Lưu ý**: Base model perplexity không được đo riêng do constraint VRAM trên T4 khi reload model nhiều lần. Perplexity của r=8 (4.65) là tham chiếu tốt nhất.

**Quan sát nổi bật**:
- r=8 đạt perplexity thấp nhất (4.65), tốt hơn r=16 và hơn hẳn r=64
- r=64 có perplexity cao nhất (8.86) — dấu hiệu rõ ràng của **overfitting** trên dataset nhỏ (180 samples)
- Training time gần như bằng nhau (~16–17 phút) — rank không ảnh hưởng đáng kể đến tốc độ train
- VRAM tăng theo rank: r=64 tốn thêm ~1.8 GB so với r=16

---

## 3. Loss Curve Analysis

Do T4 không đủ VRAM để chạy `eval_strategy` trong lúc training, chỉ có **training loss** được log theo từng step. Eval perplexity được tính riêng sau khi train xong mỗi adapter.

**Quan sát từ training loss (train loss cuối epoch 10)**:

- r=8 và r=16: training loss giảm đều và ổn định qua 10 epochs
- r=64: training loss cũng giảm, nhưng kết quả eval perplexity tệ hơn nhiều → model **overfit** vào training set

**Kết luận về overfitting**: Có xảy ra overfitting với r=64 trên dataset 180 samples. Với số lượng trainable parameters gấp ~8 lần r=8 (23.6M vs 2.9M params), mô hình r=64 có đủ capacity để "nhớ" training set thay vì học generalizable patterns. Đây là ví dụ điển hình của việc chọn rank quá cao cho một dataset nhỏ.

r=8 và r=16 ít có dấu hiệu overfitting hơn, với perplexity ở mức hợp lý (4.65 và 5.36).

---

## 4. Qualitative Comparison (5 examples)

So sánh output của **Base model** (Qwen3-4B chưa fine-tune) vs **Fine-tuned r=16**.

### Example 1

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

| | Output (trích) |
|---|---|
| **Base** | *"Máy học là một phần của học máy mà máy có thể tự cải thiện hiệu suất mà không cần được lập trình rõ ràng..."* |
| **Fine-tuned** | *"Trí tuệ nhân tạo là ngành nghiên cứu về máy tạo ra hành động và học thuật. Máy học là một loại trí tuệ nhân tạo, cho phép máy tự cải thiện hiệu suất khi tiếp xúc với dữ liệu..."* |

**Nhận xét**: Cả hai đều giải thích được khái niệm cơ bản. Fine-tuned bắt đầu bằng bối cảnh AI rộng hơn rồi mới thu hẹp vào ML — cấu trúc rõ ràng hơn. **Slight improvement**.

---

### Example 2

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

| | Output (trích) |
|---|---|
| **Base** | Dùng tên biến `res`, in kèm input prompt |
| **Fine-tuned** | Dùng tên biến `fib_sequence` (rõ nghĩa hơn), code sạch hơn, có comment ví dụ |

**Nhận xét**: Logic tương đương, nhưng fine-tuned có naming convention tốt hơn và format gọn hơn. **Minor improvement**.

---

### Example 3

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

| | Output (trích) |
|---|---|
| **Base** | Liệt kê: Tính trực quan, Tính tiếp cận, Tính nhất quán... (theo hướng kỹ thuật) |
| **Fine-tuned** | Liệt kê: Người dùng trung tâm, Tính nhất quán... (theo hướng UX-first) |

**Nhận xét**: Fine-tuned ưu tiên "Người dùng trung tâm" làm nguyên tắc số 1 — phù hợp hơn với tiêu chuẩn UX hiện đại. **Improved**.

---

### Example 4

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

| | Output (trích) |
|---|---|
| **Base** | *"LoRA (Lo-Rank Adaptation) là kỹ thuật làm nhỏ kích thước đầu ra..."* — giải thích sai cơ chế |
| **Fine-tuned** | *"LoRA, hay Loại hóa Kích thước Rút gọn..."* — tên dịch sai, nhưng giải thích cơ bản hơn |

**Nhận xét**: Cả hai đều có lỗi về thuật ngữ kỹ thuật (dataset training không có ground truth chính xác về topic này). **Same / slightly degraded** — fine-tuning không cải thiện factual accuracy khi training data không cover topic.

---

### Example 5

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

| | Output (trích) |
|---|---|
| **Base** | Giải thích 3 khái niệm tuần tự, khá rõ ràng |
| **Fine-tuned** | Có hiện tượng lặp từ ("chỉnh sửa các đầu vào đầu vào") — nhỏ nhưng đáng chú ý |

**Nhận xét**: Fine-tuned có dấu hiệu repetition nhỏ ở câu đầu. **Slightly degraded** — có thể do dataset có một vài example với pattern lặp lại.

---

**Tổng kết qualitative**:

| Prompt | Base | Fine-tuned (r=16) | Verdict |
|--------|------|-------------------|---------|
| ML explanation | Decent | More structured | ✅ Improved |
| Fibonacci code | OK | Cleaner naming | ✅ Minor improvement |
| UI/UX principles | OK | More user-centric | ✅ Improved |
| LoRA vs QLoRA | Inaccurate | Also inaccurate | ≈ Same |
| PE / RAG / FT | Clear | Slight repetition | ⚠️ Minor degradation |

---

## 5. Conclusion về Rank Trade-off

Kết quả thực nghiệm trên dataset Vietnamese Alpaca 180 samples với Qwen3-4B cho thấy một pattern rõ ràng và có ý nghĩa thực tiễn quan trọng: **rank thấp hơn không đồng nghĩa với kết quả tệ hơn**.

**r=8 cho ROI tốt nhất** trên dataset này vì 3 lý do. Thứ nhất, với chỉ 180 training samples, model có rank cao (r=64, ~23M trainable params) sẽ overfit — model "nhớ" training data thay vì học generalizable patterns về ngôn ngữ và format. Điều này được xác nhận qua perplexity: r=64 đạt 8.86 trong khi r=8 chỉ 4.65. Thứ hai, sự khác biệt về training time gần như không đáng kể (16.63 vs 17.14 phút — chưa đến 3%). Thứ ba, r=8 tiết kiệm VRAM đáng kể so với r=64 (10.97 vs 12.01 GB), giúp có thêm headroom cho batch size lớn hơn hoặc sequence length dài hơn.

**Diminishing returns** xuất hiện rất sớm trong thí nghiệm này. Từ r=8 lên r=16, số params tăng gấp đôi nhưng perplexity lại *tệ hơn* (5.36 vs 4.65). Từ r=16 lên r=64, params tăng 4×, VRAM tăng ~1.8 GB, nhưng perplexity tệ hơn nhiều (8.86). Điểm diminishing returns đã vượt qua ngay từ r=8 → r=16, nghĩa là ngưỡng tối ưu của rank với dataset 180 samples nằm ở r ≤ 8.

**Recommendation cho production**: Với dataset nhỏ (<500 samples), chọn **r=8 hoặc r=16**. Rule of thumb: dataset size / rank ≥ 50 để tránh overfitting. Nếu muốn tăng capacity mà không tăng rank, tốt hơn là tăng số lượng và chất lượng data, hoặc mở rộng `target_modules` sang các layer khác (q/k/v/o + gate/up/down). Rank cao (r=64+) chỉ có lợi khi có dataset lớn (>10k samples) và task đòi hỏi style shift rất mạnh.

---

## 6. What I Learned

- **Rank không phải càng cao càng tốt**: Với dataset nhỏ, rank cao gây overfitting rõ ràng. Đây là bài học quan trọng nhất — intuition "nhiều params = học tốt hơn" không áp dụng khi data không đủ lớn để regularize.

- **QLoRA cho phép fine-tune model 4B trên T4 miễn phí**: Toàn bộ thí nghiệm với 3 adapter chỉ tốn ~$0.30 và 50 phút. Điều này xác nhận rằng QLoRA đã thực sự democratize fine-tuning — không còn cần GPU đắt tiền để customize LLM cho domain cụ thể.

- **Qualitative evaluation quan trọng không kém perplexity**: Perplexity chỉ đo distribution fit trên eval set; nó không nói lên được model có thực sự *trả lời tốt hơn* không. Example 4 (LoRA vs QLoRA) cho thấy fine-tuning không thể bổ sung knowledge mà training data không có — đây là lý do RAG tồn tại song song với fine-tuning.

---

## Links

- **HuggingFace Hub (r=16 adapter)**: https://huggingface.co/NLT22/qwen3-4b-vi-lab21-r16
- **GitHub repo**: https://github.com/NLT22/lab21_2A202600174
