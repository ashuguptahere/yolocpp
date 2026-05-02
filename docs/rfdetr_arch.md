# RF-DETR 1.6.5 — architecture spec (source of truth for #65D..L)

This document captures the ground-truth architecture of the upstream
[`rfdetr==1.6.5`](https://pypi.org/project/rfdetr/1.6.5/) checkpoints
that we target for `--mode predict / val / train / export` of the
`rfdetr-*` model family. It exists because **the original `#65`
scaffold (0.31.0..0.37.0) was designed against a generic DETR /
LW-DETR shape and does NOT match upstream RF-DETR**; the rewrite
under #65A2..F2 (planned across follow-up sessions) replaces each
module to match the spec below.

The numbers were dumped directly from the 12 official `.pth` /
`.pt` files via `torch.load(..., weights_only=False)['model']`.
Inventories live (uncommitted) under
`/tmp/yolocpp_parity/rfdetr_dumps/*.json`.

## Variants — twelve total (5 detect + 7 segment)

| variant       | resolution | patch | bb_grid | bb_embed | hidden | dec_layers | num_queries | group_detr | refpoint  |
|---------------|-----------:|------:|--------:|---------:|-------:|-----------:|------------:|-----------:|-----------|
| nano          |   384      |  16   | 24×24   |   384    |   256  |    2       |    300      |    13      | 3900×4    |
| small         |   512      |  16   | 32×32   |   384    |   256  |    3       |    300      |    13      | 3900×4    |
| medium        |   576      |  16   | 36×36   |   384    |   256  |    4       |    300      |    13      | 3900×4    |
| base          |   560      |  14   | 37×37   |   384    |   256  |    3       |    300      |    13      | 3900×4    |
| large         |   704      |  16   | 37×37   |   768    |   384  |    3       |    300      |    13      | 3900×4    |
| seg-nano      |   368      |  16   | 26×26   |   384    |   256  |    4       |    100      |    13      | 1300×4    |
| seg-small     |   512      |  16   | 32×32   |   384    |   256  |    4       |    100      |    13      | 1300×4    |
| seg-medium    |   576      |  16   | 36×36   |   384    |   256  |    5       |    200      |    13      | 2600×4    |
| seg-large     |   672      |  16   | 42×42   |   384    |   256  |    5       |    300      |    13      | 3900×4    |
| seg-xlarge    |   624      |  12   | 52×52   |   384    |   256  |    6       |    300      |    13      | 3900×4    |
| seg-xxlarge   |   768      |  12   | 64×64   |   384    |   256  |    6       |    300      |    13      | 3900×4    |
| seg-preview   |   432      |  12   | 36×36   |   384    |   256  |    4       |    200      |    13      | 2600×4    |

**Universal constants (all 12 variants):**
- Backbone: `dinov2_windowed_small` — 12 transformer blocks, embed=384,
  6 attention heads, `mlp_ratio=4`, layer_scale1/layer_scale2 trainable
  scalars per block, separate Q/K/V projection (NOT fused).
- Backbone position-embedding tensor: `[1, bb_tokens, embed]` where
  `bb_tokens = 1 (cls) + bb_grid² × num_grids_per_window`. The
  position embedding is interpolated at inference time to match the
  actual input resolution.
- Projector neck: CSP-style 1-stage block (`projector.stages.0.0.cv1`,
  `cv2`, `m.<i>.cv1`, `cv2`) with batchnorm. Single output level
  (`P4`).
- Self-attention: fused QKV via `in_proj_weight [3·hidden, hidden]`
  + `out_proj`; `sa_nheads=8`.
- Cross-attention: single-level deformable (`num_levels=1`),
  `dec_n_points=2`, `ca_nheads=16`. Output dims:
  - `sampling_offsets [ca_nheads × levels × points × 2 = 64, hidden]`
    for hidden=256 (or 384 for large, with `ca_nheads=24` →
    24×1×8×2=384).
  - `attention_weights [ca_nheads × levels × points = 32, hidden]`.
  - `value_proj`, `output_proj`: `[hidden, hidden]` linears.
- FFN: `linear1 [2048, hidden]` + `linear2 [hidden, 2048]` (mlp
  ratio = 8 for hidden=256, ≈ 5.3 for hidden=384). GELU activation.
- Three LayerNorms per decoder layer (`norm1` after self-attn,
  `norm2` after cross-attn, `norm3` after FFN).
- Heads:
  - `bbox_embed.layers.{0,1,2}` — single shared 3-layer MLP
    `[hidden→hidden→hidden→4]`, applied per decoder layer with
    iterative refinement.
  - `class_embed` — single shared `Linear(hidden, 91)` (90 COCO
    classes + 1 background slot in the upstream training schema).
  - `refpoint_embed` — learnable initial reference points,
    `[num_queries × group_detr, 4]`. At inference, only the first
    `num_queries` rows are used; the remaining `group_detr−1` × N
    rows are training-time auxiliary groups.
  - `query_feat` — learnable query content embeddings,
    `[num_queries × group_detr, hidden]`.
- Two-stage encoder-output: `transformer.enc_output`,
  `transformer.enc_output_norm` (1×1 conv-equivalents on backbone
  taps × 13 = `13 × 2 = 26` keys), `transformer.enc_out_class_embed.0..12`
  (13 cls heads), `transformer.enc_out_bbox_embed.0..12` (13 bbox
  MLPs `hidden→hidden→hidden→4`, 78 keys = 13 × 6).

## State-dict layout (top-level prefixes)

```
backbone.0.encoder.encoder.embeddings.{cls_token, mask_token,
                                       position_embeddings,
                                       patch_embeddings.projection.{weight,bias}}
backbone.0.encoder.encoder.encoder.layer.<i>.{
    norm1.{weight,bias},
    attention.attention.{query,key,value}.{weight,bias},
    attention.output.dense.{weight,bias},
    layer_scale1.lambda1,
    norm2.{weight,bias},
    mlp.fc{1,2}.{weight,bias},
    layer_scale2.lambda1,
}                                  # 12 layers, 23 keys/layer = 276
backbone.0.encoder.encoder.layernorm.{weight,bias}
backbone.0.projector.stages.0.0.{
    cv1.{conv.weight, bn.{weight,bias}},
    cv2.{conv.weight, bn.{weight,bias}},
    m.<i>.{cv1.{...}, cv2.{...}},
}
backbone.0.projector.stages.0.1.{weight,bias}            # final BN

transformer.decoder.layers.<j>.{
    self_attn.{in_proj_weight, in_proj_bias,
               out_proj.{weight, bias}},
    norm1.{weight,bias},
    cross_attn.{sampling_offsets.{weight,bias},
                attention_weights.{weight,bias},
                value_proj.{weight,bias},
                output_proj.{weight,bias}},
    linear1.{weight,bias},
    linear2.{weight,bias},
    norm2.{weight,bias},
    norm3.{weight,bias},
}                                  # dec_layers, 22 keys/layer
transformer.enc_output.<k>.{...}                # k ∈ 0..12
transformer.enc_output_norm.<k>.{weight,bias}
transformer.enc_out_class_embed.<k>.{weight,bias}
transformer.enc_out_bbox_embed.<k>.layers.{0,1,2}.{weight,bias}

class_embed.{weight,bias}                         # final shared cls head
bbox_embed.layers.{0,1,2}.{weight,bias}           # final shared bbox MLP
refpoint_embed.weight                             # [Q·G, 4]
query_feat.weight                                 # [Q·G, hidden]

# Segment-only (added to the above):
segmentation_head.bias
segmentation_head.spatial_features_proj.{0.weight, 1.{weight,bias}}
segmentation_head.query_features_proj.{0.weight, 1.{weight,bias}}
segmentation_head.query_features_block.layers.{0,1,2}.{weight,bias}
segmentation_head.blocks.<m>.{...}
```

## Inference flow

1. Letterbox + normalise input image to `[B, 3, R, R]` where
   `R = resolution` (per-variant — see table).
2. Backbone forward → 4 ViT tap features at indices
   `out_feature_indexes` (one of `[3,6,9,12]` or `[2,5,8,11]`)
   shaped `[B, 384, R/patch, R/patch]`.
3. Projector consumes the LAST tap (P4 only — `projector_scale=['P4']`)
   → `[B, hidden, R/patch, R/patch]`.
4. Two-stage encoder predicts cls + bbox **per backbone tap** over
   the 13 layer-aware heads, picks top-`num_queries` proposals as
   the initial reference points (overwriting `refpoint_embed`'s
   first group at inference; `query_feat` is added).
5. Decoder iteratively refines `(reference_points, query_feat)` for
   `dec_layers` layers, each emitting `(cls_logits[B,Q,91],
   bbox_unact[B,Q,4])`. Iterative bbox refinement = next-layer
   reference = sigmoid(prev_unact + bbox_embed(prev_query)).
6. At eval: only the LAST decoder layer's outputs are used.
   `top-K = num_select = num_queries` highest-scoring queries; cls
   sigmoid; bbox sigmoid → cxcywh in [0,1] → xyxy in pixel coords
   via image dims.

## Differences from current `#65A..G` scaffold

| Slice  | Scaffold (0.31.0..0.37.0) | Reality (1.6.5) | Required change |
|--------|---------------------------|-----------------|-----------------|
| Backbone (#65A) | Per-scale LW-DETR / DINOv2-Large with embed_dim differing per variant; fused QKV; depths = 6/8/10/12/24 | All variants share `dinov2_windowed_small` (12 blocks, embed=384, separate Q/K/V), large is `dinov2_windowed_base` (12 blocks, embed=768) | Replace with HF DINOv2 layout matching `backbone.0.encoder.encoder.encoder.layer.<i>.attention.attention.{query,key,value}` + `layer_scale{1,2}.lambda1` |
| Projector | None | CSP-style block on backbone last tap, with batchnorm | Add `rfdetr_projector.{hpp,cpp}` |
| Encoder (#65B) | Multi-scale deformable transformer, dedicated module | Does not exist as a separate module — encoder predictions live in `transformer.enc_output[_norm]` 1×1 layers + `enc_out_{class,bbox}_embed.<k>` heads on the backbone taps | Delete `rfdetr_encoder.{hpp,cpp}`; add a small `EncoderOutput` module |
| Decoder (#65C) | Three-LN layer with vanilla MHA self-attn + multi-level (3) deformable cross-attn, separate per-layer cls/bbox heads | Three-LN layer with **fused-QKV** self-attn + **single-level** deformable cross-attn (`num_levels=1, num_points=2`), **shared** cls/bbox heads with iterative bbox refinement | Rewrite layer + add iterative refinement |
| Loss (#65F) | DETR set loss (focal+L1+GIoU) over ALL queries | Same loss family, but with **group_detr=13** auxiliary query groups during train | Add group-DETR support (only first group used at eval) |
| Trainer (#65G) | Generic | + group_detr handling, cosine LR, AMP toggle | Mostly compatible |
| Predict (#65E) | NMS-free top-K decode | Same | OK as-is |

## Reference

- Upstream sources (read-only):
  - `rfdetr/models/backbone/dinov2_with_windowed_attn.py` (1296 LOC)
  - `rfdetr/models/backbone/projector.py` (317 LOC)
  - `rfdetr/models/backbone/backbone.py` (207 LOC)
  - `rfdetr/models/lwdetr.py` (515 LOC)
  - `rfdetr/models/transformer.py` (677 LOC)
- Pretrained weights (URLs from `rfdetr.assets.ModelWeights`):
  ```
  rf-detr-{nano,small,medium,base,large}.pth
  rf-detr-seg-{nano,small,medium,large,xlarge,xxlarge,preview}.pt
  ```
- License: Apache-2.0 (compatible with our DEPS.md whitelist).
