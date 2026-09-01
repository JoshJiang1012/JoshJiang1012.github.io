# Lexa AsymCompute

> An evidence-aware toolkit for modelling and optimizing **asymmetric compute** across GPUs, CPUs, system RAM, NVMe, accelerators and edge nodes.

[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10--3.12-blue)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Evidence-aware](https://img.shields.io/badge/data-observed%20%7C%20analytical-orange)](docs/DATA_PROVENANCE.md)

Lexa AsymCompute is not tied to one model, one GPU or one workload. It treats a machine or edge fleet as a set of unequal compute tiers and answers a practical question:

> **Which fraction of a workload should run on each tier so that latency, memory, energy and transfer constraints are jointly minimized?**

The toolkit includes:

- a general divisible-workload model based on FLOPs, memory traffic, resident capacity, transfers and synchronization;
- GPU/CPU/RAM/NVMe or multi-node placement optimization;
- an asymmetric-compute index derived from Jain's fairness measure;
- energy-aware objective weighting;
- a sparse-routing specialization for workloads that activate only part of a large parameter or operator pool;
- privacy-safe route-trace validation and chronological cache holdout evaluation;
- observed VM and synthetic-pipeline measurements kept strictly separate from analytical estimates;
- standard-library-only Python code, tests, schemas and reproducible examples.

---

## Why asymmetric compute?

Real systems are not homogeneous:

```text
GPU       high arithmetic throughput + high local bandwidth + limited VRAM
CPU       flexible control flow + moderate compute + large address space
RAM       larger capacity + lower bandwidth than VRAM
NVMe      very large capacity + high latency + no direct arithmetic
Edge node low power + local data + intermittent network
Cloud GPU high throughput + remote transfer cost
```

A symmetric scheduler assumes every worker is interchangeable. Lexa AsymCompute instead models the hardware differences explicitly:

```text
mandatory work stays where it is efficient or required
parallel work is divided by measured/effective service rate
state is placed under capacity limits
output is transferred through explicit links
all concurrent paths meet at a critical-path barrier
```

This is useful when a workload is too large for one tier, when moving everything to the fastest device is impossible, or when the fastest device is already occupied by mandatory work.

---

## Evidence status

The repository distinguishes four evidence classes:

| Class | Meaning |
|---|---|
| `observed_measurement` | Directly measured in a declared environment |
| `observed_synthetic_harness` | Real runtime measurement on synthetic input |
| `analytical_estimate` | Equation output from declared assumptions |
| `analytical_proxy` | A proxy such as active-weight traffic, not direct energy/FLOPs |

The project never relabels an analytical estimate as a hardware benchmark.

### Observed in the isolated validation VM

The following values were measured by `scripts/benchmark_host.py` on the validation environment, using 64 MiB buffers and five repetitions:

| Measurement | Observed value |
|---|---:|
| Logical CPUs visible | 5 |
| Memory visible | 5.809 GiB |
| NVIDIA GPU visible | No |
| Python `bytearray` copy, median | **11.672 GB/s** |
| SHA-256 throughput, median | **1.708 GB/s** |
| Temporary filesystem read, median | **2.419 GB/s** |
| Temporary filesystem write, median | **4.964 GB/s** |

These values characterize the VM and Python harness only. They do **not** represent a desktop GPU, CUDA kernel or neural-network inference runtime.

### Observed synthetic route-pipeline stress run

A seeded synthetic trace with 4,096 units, 36 layers and top-4 routing generated and analyzed:

| Measurement | Observed value |
|---|---:|
| Routing events | 147,456 |
| JSONL size | 27,227,837 bytes |
| Generation wall time | 3.39 s |
| Generation throughput | 43,497 events/s |
| Generation peak RSS | 90.55 MiB |
| Analysis wall time | 1.76 s |
| Analysis throughput | 83,782 events/s |
| Input processing rate | 15.47 MB/s |
| Analysis peak RSS | 164.10 MiB |
| Geometry validation | Pass |
| 70/30 holdout selection hit rate, cache 18 | 61.19% |
| Full top-4 row hit rate | 12.13% |

The timings are real measurements of the trace implementation, but the routing pattern is synthetic and is **not model inference throughput**. Raw summaries are under [`data/observed/`](data/observed/).

---

## Quick start

```bash
git clone <repository-url>
cd Lexa-AsymCompute
python -m pip install -e .
python -m unittest discover -s tests -v
```

Optimize the included heterogeneous edge workload:

```bash
lexa-asymcompute optimize \
  --workload data/workloads/edge_pipeline_example.json \
  --hardware data/hardware/desktop_heterogeneous_example.json \
  --step 0.01
```

Estimate a specific placement:

```bash
lexa-asymcompute estimate \
  --workload data/workloads/edge_pipeline_example.json \
  --hardware data/hardware/desktop_heterogeneous_example.json \
  --placement '{"gpu":0.80,"cpu_ram":0.20}'
```

Evaluate a sparse-workload reference profile:

```bash
lexa-asymcompute moe-savings \
  --model data/workloads/sparse_moe_reference.json

lexa-asymcompute moe-optimize \
  --model data/workloads/sparse_moe_reference.json \
  --hardware data/hardware/moe_three_tier_assumed.json
```

Generate and analyze a privacy-safe synthetic trace:

```bash
python scripts/generate_trace_demo.py /tmp/routes.jsonl \
  --tokens 256 --layers 12 --experts 64 --top-k 4

lexa-asymcompute trace \
  --trace /tmp/routes.jsonl \
  --layers 12 \
  --top-k 4 \
  --experts-per-layer 64 \
  --cache 12 \
  --calibration 0.70
```

---

# Complete mathematical model implemented by v2.0

## 1. Sets and indices

Let:

- \(\mathcal{D}\): set of devices or tiers;
- \(\mathcal{L}\): set of directed inter-device links;
- \(d\in\mathcal{D}\): one device;
- \((d,a)\in\mathcal{L}\): a link from device \(d\) to aggregation device \(a\);
- \(x_{d}\in[0,1]\): fraction of divisible work assigned to device \(d\).

The fractional placement must satisfy:

$$
\sum_{d\in\mathcal{D}_{e}}x_{d}=1,
\qquad
x_{d}=0\;\;\forall d\notin\mathcal{D}_{e},
$$

where \(\mathcal{D}_{e}\subseteq\mathcal{D}\) is the eligible-device set.

---

## 2. Effective device capability

For each device \(d\), define:

- \(P_{d}^{\mathrm{peak}}\): peak arithmetic rate in FLOP/s;
- \(B_{d}^{\mathrm{peak}}\): peak local-memory bandwidth in byte/s;
- \(\eta_{d}^{P}\in(0,1]\): effective compute efficiency;
- \(\eta_{d}^{B}\in(0,1]\): effective bandwidth efficiency;
- \(\ell_{d}\): fixed per-stage device latency.

Effective rates are:

$$
P_{d}=\eta_{d}^{P} P_{d}^{\mathrm{peak}},
$$

$$
B_{d}=\eta_{d}^{B} B_{d}^{\mathrm{peak}}.
$$

Efficiencies should be calibrated from a representative kernel whenever possible. A vendor peak multiplied by an assumed efficiency remains analytical input, not an observed result.

---

## 3. Work decomposition

A stage is divided into:

- \(F_{p}\): divisible arithmetic work in FLOPs;
- \(Q_{p}\): divisible local-memory traffic in bytes;
- \(R_{p}\): divisible resident state in bytes;
- \(F_{d}^{m}\): mandatory arithmetic work fixed to device \(d\);
- \(Q_{d}^{m}\): mandatory memory traffic fixed to device \(d\);
- \(R_{d}^{m}\): mandatory resident state fixed to device \(d\);
- \(F_{s},Q_{s}\): serial arithmetic and memory work;
- \(Q_{o}\): total output bytes returned to an aggregation device;
- \(\tau_{\mathrm{sync}}\): synchronization overhead.

The assigned work on device \(d\) is:

$$
F_{d}=x_{d}F_{p}+F_{d}^{m},
$$

$$
Q_{d}=x_{d}Q_{p}+Q_{d}^{m},
$$

$$
R_{d}=x_{d}R_{p}+R_{d}^{m}.
$$

---

## 4. Capacity constraint

For device memory capacity \(C_{d}\):

$$
R_{d}\le C_{d}
\qquad\forall d\in\mathcal{D}.
$$

A placement violating this inequality is infeasible and is rejected rather than assigned an optimistic penalty.

---

## 5. Device service time

Under a roofline-style overlap assumption, arithmetic and local-memory work compete on the slower of the two paths:

$$
T_{d}^{\mathrm{compute}}=\frac{F_{d}}{P_{d}},
$$

$$
T_{d}^{\mathrm{memory}}=\frac{Q_{d}}{B_{d}},
$$

$$
T_{d}^{\mathrm{service}}=\max\left(T_{d}^{\mathrm{compute}},T_{d}^{\mathrm{memory}}\right)+\ell_{d}.
$$

The arithmetic intensity of device shard \(d\) is:

$$
I_{d}=\frac{F_{d}}{Q_{d}}\quad\text{FLOP/byte}.
$$

The device balance point is:

$$
I_{d}^{*}=\frac{P_{d}}{B_{d}}.
$$

Therefore:

- \(I_{d}<I_{d}^{*}\): shard is bandwidth-bound;
- \(I_{d}>I_{d}^{*}\): shard is compute-bound;
- \(I_{d}\approx I_{d}^{*}\): shard is near the roofline knee.

---

## 6. Link transfer time

For link bandwidth \(B_{\mathrm{d,a}}^{\mathrm{link}}\), link latency \(\ell_{\mathrm{d,a}}^{\mathrm{link}}\), and output fraction \(x_{d}Q_{o}\):

$$
T_{\mathrm{d,a}}^{\mathrm{transfer}}
=
\ell_{\mathrm{d,a}}^{\mathrm{link}}
+
\frac{x_{d}Q_{o}}{B_{\mathrm{d,a}}^{\mathrm{link}}}.
$$

The implementation adds transfer time to the emitting device path:

$$
T_{d}^{\mathrm{path}}=T_{d}^{\mathrm{service}}+T_{\mathrm{d,a}}^{\mathrm{transfer}}.
$$

When \(d=a\), the transfer term is zero.

---

## 7. Parallel critical path

Concurrent device paths meet at a barrier. The parallel stage is limited by the slowest path:

$$
T_{\mathrm{parallel}}=\max_{d\in\mathcal{D}_{e}}T_{d}^{\mathrm{path}}.
$$

This is the central asymmetric-compute rule: the goal is not equal work, but approximately equal **completion time**.

At an unconstrained optimum among active devices:

$$
T_{i}^{\mathrm{path}}\approx T_{j}^{\mathrm{path}}
\qquad\forall i,j\text{ with }x_{i},x_{j}>0.
$$

---

## 8. Serial work

For serial device \(s\):

$$
T_{\mathrm{serial}}
=
\max\left(
\frac{F_{s}}{P_{s}},
\frac{Q_{s}}{B_{s}}
\right)+\ell_{s}.
$$

---

## 9. Total latency and throughput

The implemented single-stage model is:

$$
\boxed{
T_{\mathrm{total}}
=
T_{\mathrm{serial}}
+
T_{\mathrm{parallel}}
+
\tau_{\mathrm{sync}}
}
$$

For one completed workload unit per stage:

$$
\boxed{
\Theta=\frac{1}{T_{\mathrm{total}}}
}
$$

where \(\Theta\) is units/s and all time terms are expressed in seconds.

For a chain of stages \(k=1,\ldots,K\) without pipeline overlap:

$$
T_{\mathrm{chain}}=\sum_{k=1}^{K}T_{\mathrm{total,k}}.
$$

For a steady-state pipeline with independent buffers:

$$
T_{\mathrm{pipeline,period}}=\max_{k} T_{\mathrm{total,k}},
$$

$$
\Theta_{\mathrm{pipeline}}=\frac{1}{\max_{k} T_{\mathrm{total,k}}}.
$$

---

## 10. Energy model

For device power \(W_{d}\) in watts:

$$
E_{d}^{\mathrm{compute}}=W_{d}T_{d}^{\mathrm{service}}.
$$

For link energy intensity \(\epsilon_{\mathrm{d,a}}\) in joule/byte:

$$
E_{\mathrm{d,a}}^{\mathrm{transfer}}=\epsilon_{\mathrm{d,a}}x_{d}Q_{o}.
$$

Total energy is:

$$
\boxed{
E_{\mathrm{total}}
=
E_{\mathrm{serial}}
+
\sum_{d} E_{d}^{\mathrm{compute}}
+
\sum_{(d,a)\in\mathcal{L}}E_{\mathrm{d,a}}^{\mathrm{transfer}}
}
$$

The default examples omit link-energy constants unless measured; in that case transfer energy is zero in the estimate and must not be interpreted as physically zero.

---

## 11. Optimization objective

Lexa AsymCompute performs constrained grid search over the fractional simplex. The objective is:

$$
\boxed{
J(x)
=
\lambda_{T}T_{\mathrm{total}}(x)
+
\lambda_{E}E_{\mathrm{total}}(x)
}
$$

subject to:

$$
\sum_{d}x_{d}=1,
$$

$$
x_{d}\ge0,
$$

$$
R_{d}\le C_{d},
$$

$$
x_{d}=0\quad\forall d\notin\mathcal{D}_{e}.
$$

The default is latency-only optimization:

$$
\lambda_{T}=1,
\qquad
\lambda_{E}=0.
$$

---

## 12. Closed-form bandwidth-only special case

Consider a partitionable traffic amount \(A\), mandatory traffic \(M_{d}\) on each active tier, ideal overlap and no fixed latency. Let effective bandwidth be \(B_{d}\). Each path time is:

$$
T_{d}=\frac{M_{d}+x_{d}A}{B_{d}}.
$$

At the balanced optimum, active paths finish together at \(T^{*}\):

$$
M_{d}+x_{d}A=B_{d}T^{*}.
$$

Summing over active tiers and using \(\sum_{d}x_{d}=1\):

$$
\sum_{d}M_{d}+A=T^{*}\sum_{d}B_{d}.
$$

Thus:

$$
\boxed{
T^{*}=\frac{A+\sum_{d}M_{d}}{\sum_{d}B_{d}}
}
$$

and:

$$
\boxed{
x_{d}^{*}
=
\frac{B_{d} T^{*} - M_{d}}{A}
}
$$

for tiers with positive \(x_{d}^{*}\). Negative shares are clamped to zero and the system is re-solved on the remaining active set. Capacity, fixed latency, transfer and discrete placement generally require numerical optimization.

---

## 13. Asymmetry index

For a positive capability vector \(c=(c_{1},\ldots,c_{n})\), Jain's fairness index is:

$$
J(c)=\frac{\left(\sum_{i=1}^{n}c_{i}\right)^{2}}{n\sum_{i=1}^{n}c_{i}^{2}}.
$$

Lexa defines normalized capability asymmetry as:

$$
\boxed{
A(c)=1-J(c)
}
$$

Properties:

$$
0\le A(c)<1.
$$

- \(A=0\): equal capabilities;
- larger \(A\): increasingly unequal tiers;
- the metric can be applied separately to compute, bandwidth, capacity, latency or energy efficiency.

A system can be compute-symmetric but bandwidth-asymmetric, so multiple capability vectors should be reported.

---

## 14. Speedup and latency reduction

For baseline latency \(T_{b}\) and optimized latency \(T_{o}\):

$$
\boxed{S=\frac{T_{b}}{T_{o}}}
$$

$$
\boxed{R_{T}=1-\frac{T_{o}}{T_{b}}=1-\frac{1}{S}}
$$

where \(S\) is speedup and \(R_{T}\) is fractional latency reduction.

Throughput gain is:

$$
G_\Theta=\frac{\Theta_{o}-\Theta_{b}}{\Theta_{b}}=S-1
$$

when one workload unit is completed per latency interval.

---

## 15. Amdahl-style upper bound

Let \(f_{s}\) be the strictly serial fraction and \(f_{k}\) the fractions accelerated by mechanisms with speedups \(s_{k}\). Let \(\tau\) be normalized scheduling/transfer overhead. Then:

$$
\boxed{
S_{\mathrm{max}}
=\frac{1}{f_{s}+\sum_{k}\frac{f_{k}}{s_{k}}+\tau}
}
$$

This prevents an optimizer from claiming unlimited benefit when serial control, transfer or synchronization dominates.

---

## 16. Sparse activation specialization

Some workloads have a large operator or parameter pool but activate only a subset per unit. Let:

- \(N\): total operators/experts per layer;
- \(K\): active operators/experts per unit;
- \(L\): layers;
- \(S_{e}\): bytes per operator/expert;
- \(D\): mandatory dense traffic per unit.

Active sparse traffic is:

$$
A=LKS_{e}.
$$

All-operator traffic would be:

$$
A_{\mathrm{full}}=LNS_{e}.
$$

Sparse activation ratio:

$$
r_{e}=\frac{K}{N}.
$$

Sparse work proxy avoided:

$$
\boxed{1-r_{e}=1-\frac{K}{N}}
$$

Active total traffic:

$$
Q_{\mathrm{active}}=D+A.
$$

All-operator total traffic:

$$
Q_{\mathrm{full}}=D+A_{\mathrm{full}}.
$$

Traffic proxy reduction:

$$
\boxed{
R_{Q}=1-\frac{Q_{\mathrm{active}}}{Q_{\mathrm{full}}}
}
$$

This is a weight/operator-traffic proxy. It is not direct joules, FLOPs or wall-clock time.

---

## 17. Cache residency constraint

For item \(i\), size \(s_{i}\), device capacity \(C_{d}\), and binary residency variable \(y_{\mathrm{i,d}}\in\{0,1\}\):

$$
\sum_{i} s_{i} y_{\mathrm{i,d}}\le C_{d}.
$$

With access probability \(p_{i}\) and tier service cost \(c_{\mathrm{i,d}}\), expected access cost is:

$$
\mathbb{E}[C]
=
\sum_{i} p_{i}\sum_{d} y_{\mathrm{i,d}}c_{\mathrm{i,d}}.
$$

A simple cache-placement objective is:

$$
\min_{y}
\left[
\sum_{i} p_{i}\sum_{d} y_{\mathrm{i,d}}c_{\mathrm{i,d}}
+
\lambda_{\mathrm{churn}}\sum_{\mathrm{i,d}}|y_{\mathrm{i,d}}^{(t)}-y_{\mathrm{i,d}}^{(t-1)}|
\right].
$$

---

## 18. Chronological cache holdout

The route evaluator uses the first fraction \(q\) of token/unit indices for calibration and the remaining \(1-q\) for evaluation:

$$
\mathcal{T}_{\mathrm{cal}}=\{t:t<t_{q}\},
$$

$$
\mathcal{T}_{\mathrm{eval}}=\{t:t\ge t_{q}\}.
$$

For cache set \(C_{l}\) at layer \(l\), selection hit rate is:

$$
H_{\mathrm{selection}}
=
\frac{\sum_{\mathrm{t,l}}|S_{\mathrm{t,l}}\cap C_{l}|}{\sum_{\mathrm{t,l}}|S_{\mathrm{t,l}}|}.
$$

Full-row hit rate is:

$$
H_{\mathrm{row}}
=
\frac{\sum_{\mathrm{t,l}}\mathbf{1}[S_{\mathrm{t,l}}\subseteq C_{l}]}{|\mathcal{T}_{\mathrm{eval}}|L}.
$$

The cache is learned only from calibration data; evaluation data is not used to choose entries.

---

## 19. Critical-miss probability

If one layer/stage has independent blocking-miss probability \(p\), the probability of at least one miss across \(L\) stages is:

$$
P_{\mathrm{miss,unit}}=1-(1-p)^{L}.
$$

To meet a unit-level miss budget \(\epsilon\):

$$
1-(1-p)^{L}\le\epsilon,
$$

so:

$$
\boxed{
p\le1-(1-\epsilon)^{1/L}}
$$

For small \(p\):

$$
p\lesssim\frac{\epsilon}{L}.
$$

---

## 20. Prefetch horizon

Let item load time be \(T_{\mathrm{load}}\) and average stage budget be \(\tau_{\mathrm{stage}}\). The minimum look-ahead distance is:

$$
\boxed{
H_{\mathrm{prefetch}}
\ge
\left\lceil\frac{T_{\mathrm{load}}}{\tau_{\mathrm{stage}}}\right\rceil
}
$$

A long-tail latency budget should use a percentile such as \(T_{\mathrm{load}}^{P99}\), not only the mean:

$$
H_{\mathrm{prefetch}}^{P99}
\ge
\left\lceil\frac{T_{\mathrm{load}}^{P99}}{\tau_{\mathrm{stage}}}\right\rceil.
$$

---

## 21. Measurement uncertainty and sensitivity

If effective capability \(c_{i}\) has relative uncertainty \(u_{i}\), evaluate a lower and upper scenario:

$$
c_{i}^{-} = c_{i}(1-u_{i}),
\qquad
c_{i}^{+} = c_{i}(1+u_{i}).
$$

The reported latency interval is:

$$
T^{-} = T(c_{1}^{+},\ldots,c_{n}^{+}),
$$

$$
T^{+} = T(c_{1}^{-},\ldots,c_{n}^{-}).
$$

Local sensitivity of metric \(Y\) to parameter \(z\) is:

$$
S_{z}^{Y}
=
\frac{\partial Y}{\partial z}\frac{z}{Y}.
$$

Finite-difference approximation:

$$
S_{z}^{Y}
\approx
\frac{Y(z+\Delta z)-Y(z-\Delta z)}{2\Delta z}\frac{z}{Y(z)}.
$$

A robust publication should report sensitivity when efficiencies are assumed rather than measured.

---

## 22. Analytical reference result

The included three-tier sparse-workload example uses declared effective bandwidth assumptions:

```text
GPU effective path:   394.24 GB/s
RAM effective path:    50.00 GB/s
NVMe effective path:    5.50 GB/s
fixed overhead:          1.50 ms/unit
```

Grid search at 0.25 percentage-point resolution returns:

```text
GPU share:   68.25%
RAM share:   28.75%
NVMe share:   3.00%
latency:     12.4903 ms/unit
throughput:  80.0623 units/s
```

The declared CPU/RAM-only analytical baseline is:

```text
latency:     39.5720 ms/unit
throughput:  25.2704 units/s
```

Derived analytical comparison:

$$
S=\frac{39.5720}{12.4903}=3.1687\times,
$$

$$
R_{T}=1-\frac{12.4903}{39.5720}=68.43\%.
$$

These numbers are reproducible equation outputs from bundled assumptions, not observed device throughput.

---

## Suggested uses

Lexa AsymCompute is suitable for:

1. **Local AI inference planning** — decide which layers, operators, caches or preprocessing steps belong on accelerator, CPU or storage.
2. **Sparse model routing** — evaluate hot/cold operator placement without making the entire project model-specific.
3. **Edge-cloud partitioning** — place sensor preprocessing locally and expensive inference remotely while accounting for link latency.
4. **Video/audio pipelines** — split decode, filtering, feature extraction and inference across hardware tiers.
5. **Compilation and test farms** — assign parsing, compiling, linking and tests to unequal workers.
6. **Scientific workflows** — partition simulation kernels, checkpointing and reduction phases.
7. **Database/query execution** — place scans, filters, joins and aggregation according to bandwidth and memory limits.
8. **Robotics and embedded systems** — balance local reaction latency against remote compute capacity.
9. **Defensive security analytics** — assign parsing, static analysis, log correlation and model-assisted reasoning to appropriate tiers in authorized environments.
10. **Battery-aware scheduling** — optimize latency and energy jointly by raising \(\lambda_{E}\).
11. **Storage-backed execution research** — quantify when a cold NVMe tier helps capacity but harms tail latency.
12. **Education and architecture review** — make hidden assumptions about compute, bandwidth and transfer explicit.

Not recommended without extension:

- cycle-accurate GPU simulation;
- exact compiler/kernel benchmarking;
- hard real-time certification;
- claiming observed performance from uncalibrated peak specifications;
- treating synthetic route traces as production routing distributions.

---

## Repository layout

```text
Lexa-AsymCompute/
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── GOVERNANCE.md
├── ROADMAP.md
├── SECURITY.md
├── CITATION.cff
├── pyproject.toml
├── src/lexa_asymcompute/
│   ├── models.py          # general workload/device/link equations
│   ├── optimizer.py       # constrained simplex grid search
│   ├── metrics.py         # asymmetry, fairness, speedup
│   ├── moe.py             # optional sparse-routing specialization
│   ├── trace.py           # privacy-safe trace analysis
│   ├── provenance.py      # evidence classification contract
│   └── cli.py
├── data/
│   ├── hardware/
│   ├── workloads/
│   ├── observed/
│   ├── analytical/
│   └── schemas/
├── docs/
│   ├── MATHEMATICAL_MODEL.md
│   ├── ARCHITECTURE.md
│   ├── DATA_PROVENANCE.md
│   ├── MEASURED_DATA.md
│   ├── BENCHMARK_PROTOCOL.md
│   ├── USE_CASES.md
│   └── LIMITATIONS.md
├── examples/
├── scripts/
└── tests/
```

---

## Reproduce all included data

```bash
python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/generate_reference_data.py
PYTHONPATH=src python scripts/benchmark_host.py \
  --output data/observed/host.local.json
PYTHONPATH=src python scripts/validate_release.py
```

Observed files should be committed only with their environment, timestamp, parameters and evidence classification.

---

## Roadmap

- multi-stage DAG optimizer;
- percentile latency distributions instead of point estimates;
- device-specific kernel calibration adapters;
- Pareto frontier for latency, energy and monetary cost;
- adaptive cache policies and trace replay;
- optional mixed-integer solver backend;
- edge-node failure and intermittent-link models;
- real target-hardware contribution templates.

See [`ROADMAP.md`](ROADMAP.md).

## License

MIT. See [`LICENSE`](LICENSE).
