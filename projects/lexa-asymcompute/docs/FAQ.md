# FAQ

## Is this a hardware benchmark?

No. The engine is an analytical model. Files under `data/observed/` are measurements of explicitly declared environments; files under `data/analytical/` are equation outputs.

## Why not put everything on the GPU?

The GPU can be constrained by VRAM, mandatory dense work, transfer overhead or unsuitable control flow. The fastest placement equalizes completion time, not work fraction.

## Why use grid search?

It is dependency-free, deterministic and easy to audit. For many devices or fine resolution, a future solver backend will be more efficient.

## Can it model a non-AI workload?

Yes. The generic model requires only FLOPs, bytes, resident capacity, device rates, links and synchronization. AI/MoE is one optional specialization.

## Are synthetic route hit rates real?

They are real measurements of synthetic data, not production distributions. Use observed route metadata for a hardware/workload claim.
