# reproducing-smell 👃🎈

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
