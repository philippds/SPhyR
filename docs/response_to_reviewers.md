# Response to Reviewers

Draft. Every number below is reproducible from the repository:

```bash
python -m sphyr.evaluate_dynamic validate --subject full_easy --samples 20
python -m sphyr.evaluate_dynamic rescore --workers 8
```

---

We thank the reviewers for the criticism of our evaluation protocol. The
objection — that verification by string matching cannot establish whether a
predicted structure is physically sound — is correct, and we agree it was the
central weakness of the submission. We have replaced the protocol rather than
defended it.

## What we changed

SPhyR now evaluates completions **dynamically**, by simulating them and
re-running a real topology optimiser on the same problem:

1. **Simulation.** Each completion is turned back into the structural problem
   the sample encodes — a unit load over the `L` cells, the `S` cells clamped,
   one bilinear plane-stress finite element per grid cell, stiffness
   interpolated from density by the SIMP law — and solved for its
   **compliance**, the work done by the load and hence the inverse of
   stiffness.
2. **Re-optimisation.** A SIMP optimiser (density filter, optimality-criteria
   update, penalisation continuation, discrete swap refinement) is re-run on
   *exactly* the masked cells, with every unmasked cell pinned, under the
   reference material budget.
3. **Scoring.** A completion is scored by how close it comes to that design,
   discounted by any material it spent beyond the budget.

The physics engine is part of the released code (`src/sphyr/physics/`) and is
independent of the benchmark; its element stiffness matrix agrees with the
closed-form matrix of the standard 99/88-line topology optimisation codes to
$1.7\times10^{-16}$.

## Why the objection was well founded

Re-scoring the 24,325 valid completions in our results shows the old protocol
was not merely imprecise but systematically wrong about the property the
benchmark claims to measure:

- **72.7% of completions carry their load correctly while differing from the
  reference answer somewhere.** Under cell-by-cell comparison these are
  penalised for being different, not for being wrong.
- The converse also occurs. In one sample a model reproduces 99 of 100 cells;
  the single cell it misses is the one that was masked, and it severs the only
  path from the load to the supports. String matching scores this 99%.

Both cases are shown side by side in the new figure
(`results/plots/dynamic_evaluation/`), together with the design our optimiser
produces for the same hole and budget.

We note that the graph-connectivity metric in the submission was also not a
sound proxy for load-carrying behaviour. It disagrees with the finite element
solution on 1,079 completions: 486 that it calls connected but that cannot
transmit force — cells touching only at a corner share a single node and form a
hinge — and 593 that carry load while it calls them disconnected. It was
additionally saturated at 97–99% for the four strongest models, so it separated
almost nothing.

## The metric

For a completion with compliance $c$ and volume fraction $v$, against a
reference design of compliance $c^\star$ at budget $v_\text{ref}$:

| Metric | Definition |
| --- | --- |
| Structural efficiency | $\min(1,\ c^\star/c)$ |
| Material efficiency | $\min(1,\ v_\text{ref}/v)$ |
| **Topology score** | **their product** |

all zero when the completion carries no load.

Both factors are needed: stiffness alone is maximised by filling the grid solid,
thrift alone by building nothing. This is not only an argument in principle.
Among load-carrying completions, those spending **more** material than the
reference beat the budget-constrained reference design 68.5% of the time — as
they should, since they are spending more. At **equal** material that falls to
5.3%, and at **less** material to 0.3%. Structural efficiency on its own would
read material overspend as design quality.

## Validation of the metric

We verify that the score orders completions whose relative quality is known
beforehand (`full_easy`, 20 samples):

| Completion | Topology score |
| --- | --- |
| Reference answer | 0.93 |
| Reference answer, 2 cells flipped | 0.80 |
| Reference answer, 10 cells flipped | 0.48 |
| Grid filled solid | 0.29 |
| Grid left empty | 0.00 |

The reference answer does not score 1.0 because the optimiser generally finds a
slightly stiffer structure for the same material; that margin is exactly the
headroom a model is measured against. The optimiser is seeded with the reference
answer, so the design a completion is compared against is never worse than the
answer the dataset ships — the validation command reports this count on every
run, and it is zero.

## Effect on the reported results

The ranking changes at the top, which we take as evidence that the two protocols
measure different things:

| Model | Exact match | Topology score | Load-carrying |
| --- | --- | --- | --- |
| Gemini 2.5 Pro | 28.1% | **84.3** | 97.6% |
| Claude Opus 4 | 30.2% | **83.1** | 94.7% |
| Claude 3.7 Sonnet | 15.1% | **79.0** | 96.3% |
| GPT-4.1 | 4.3% | **72.7** | 95.6% |

Exact match ranks Claude Opus 4 above Gemini 2.5 Pro; the physics-based score
reverses them. Opus reproduces the reference string more often, Gemini builds
better structures. GPT-4.1 is the clearest case: near the bottom on exact match
at 4.3%, but 95.6% of its completions carry their load and it scores 72.7 —
above Gemini 1.5 Pro, which matches the reference string five times as often.

## A correction to the submitted results

While rebuilding the evaluation we found that the gravity direction was not
rotated along with the samples in the rotation ablation: both direction-dependent
metrics propagated force downwards even for rotated grids, whose load travels
left to right. This affected every rotation result in the submission. Corrected,
directional connectivity moves by +13 points per 100 samples on average and
force-path efficiency by −3.7. All rotation tables and figures have been
regenerated, and the conclusions of that ablation should be read from the
corrected numbers.

## Limitations

- The reference design is a strong local optimum, not a certified global one.
  Where a completion matches the reference material and is meaningfully stiffer
  than the dataset answer, it typically also beats our optimiser (5.3% of
  equal-material completions); the score clips at 1.0 in those cases, so it
  measures "at least as good as the best design we can find" rather than a
  distance to a true optimum.
- Compliance minimisation under a single static load case is one objective among
  several. Buckling, multiple load cases and stress constraints are not modelled.
- The evaluation is 2D. Extending it to the 3D subset requires hexahedral
  elements and is left to future work.
