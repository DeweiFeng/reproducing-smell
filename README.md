# reproducing-smell 👃🎈

Goal: Cracking The Olfactory Code!

1. Sensor read → raw vector S
The dataset loads each gas-nose CSV and returns a tensor `sensor` of shape `[seq_len, features]`.

2. SmellNet (LSTM) → embeddings z₁…z₅₀
In `MultimodalOdorNet`, `self.smell_enc` runs the sensor sequence through an LSTM and projects its output into a fixed embedding for each of SmellNet V0.1 50 odor channels (the zᵢ).

3. α-weights & sum → unified “odor” embedding
A learned linear layer (`smell_proj`) applies implicit αᵢ weights to those zᵢ and sums them into one fused smell vector.

4. Multimodal fusion → mixture ratios β₁…β₁₂
In parallel, Qwen encodes our image+text; all three modality embeddings (vision, text, smell) are projected into the same space, stacked, and passed through a TransformerEncoder. Finally, `output_head` Softmaxes to produce β₁…β₁₂—the mix proportions over 12 base scents.

7. playSmell(id,duration)
The script picks the βⱼ with highest score (→ `scent_id`), calls `nw.playSmell(scent_id, duration)`, and sends that hex command over serial to the NeckWear device to diffuse the predicted fragrance.

So this code exactly implements:

- Stage 1 (α) = LSTM + `smell_proj`
- Stage 2 (β) = Transformer fusion + `output_head`
- Runtime = loop over data → select top β → `playSmell`

With a properly trained `checkpoint.pt`, this runs end-to-end as drawn.

---


**Potential Failure Modes**

1. **Sensor Data Issues**

   * Missing or misnamed CSVs, malformed files.
   * Outliers or noisy readings that corrupt LSTM embeddings.

2. **Image Problems**

   * Empty or non-JPG folders, corrupted files.
   * Inconsistent transforms between training and inference.

3. **Text Description Errors**

   * Missing or malformed `long_description` in JSON.
   * Unsupported characters or overly long strings breaking the tokenizer.

4. **Model Checkpoint Mismatch**

   * `models/checkpoint.pt` absent or incompatible with current architecture.
   * PyTorch/Transformers version conflicts.

5. **Serial Handshake Failures**

   * Wrong COM port, device never responds to `getUuid()`/`wakeUp()`.
   * Timeouts too short, buffer overflows, or device resets.

6. **Inference Exceptions**

   * Type mismatches (`image`, `text`, `sensor`) causing runtime errors.
   * Out-of-memory if batches or images are too large.

7. **Flat β-Distribution**

   * Nearly uniform Softmax → unstable `argmax` between similar classes.
   * Misaligned `time.sleep(duration)` causing sync drift.

8. **Hardware Malfunctions**

   * Empty or clogged cartridges, low battery, power interruptions.

9. **Insufficient Logging & Recovery**

   * Lack of detailed logs for post-mortem debugging.
   * No retry logic for transient I/O or serial errors.

---

**Recommended Mitigations**

* **File Validation:** Assert existence and correct format of CSVs, images, JSON.
* **Robust I/O:** Wrap file and serial operations in `try/except` with retries and exponential back-off.
* **Structured Logging:** Record timestamps, batch IDs, error details.
* **Latency Monitoring:** Measure handshake and emission delays; adjust timeouts.
* **Unit Tests:** Validate dataset outputs and model shapes.
* **Hardware Checks:** Pre-run sanity tests for cartridges and power.
