# models/ — world model (Phase 1)

- `encoder.py` — 4-layer stride-2 CNN, channel LayerNorm + SiLU, flat embedding (no projection).
- `decoder.py` — mirror transposed CNN from `[h, z]`; MSE / unit-variance Gaussian (simplified vs V3).
- `rssm.py` — RSSM core: GRU deterministic state, prior `p(z|h)`, posterior `q(z|h,e)`;
  categorical (32x32, straight-through, unimix) and gaussian latents behind `latent_type`;
  `observe()` = posterior path (training), `imagine()` = prior-only path (open-loop / Phase 2);
  KL with balancing + free nats.
- `heads.py` — reward (MSE; symlog+two-hot is a TODO) and continue (BCE on `1 - terminated`).
- `world_model.py` — combines everything; masked loss with per-component metrics.

Phase 2 (actor/critic trained in imagination) lives in `agents/` — not here.
