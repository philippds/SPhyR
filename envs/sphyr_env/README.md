# SPhyR Environment

SPhyR as an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment.

An episode is one **spatial-physical reasoning** task: the environment hands
out a structural material distribution with cells masked out, the agent submits
the completed grid, and the environment scores it by **simulating it** — the
completion is solved as a linear elastic structure and a SIMP topology
optimiser is re-run on exactly the masked cells. A structure that routes the
load differently from the dataset's answer but just as efficiently scores just
as well; one that matches almost everywhere but severs the load path scores
zero.

Episodes are single-step: `reset()` poses the task, `step()` ends it.

## Grid encoding

| Token | Meaning |
| --- | --- |
| `L` | applied load |
| `S` | support |
| `V` | masked cell, for the agent to fill |
| `0`–`1` | material density (`0`/`1` on `easy` subjects, one decimal place on `hard`) |

## Usage

```python
from sphyr_env import SPhyREnv, SPhyRAction

env = SPhyREnv(base_url="http://localhost:8000")

result = env.reset(subject="10_random_cell_easy")
print(result.observation.grid)     # the masked grid
print(result.observation.prompt)   # the task, phrased for a language model

result = env.step(SPhyRAction(grid=completed_grid_text))
print(result.reward)                                 # topology score, 0..1
print(result.observation.metrics["load_carrying"])   # did it stand up?
print(result.observation.ground_truth)               # the dataset's answer

env.close()
```

In-process, without a server:

```python
from sphyr_env.server.sphyr_environment import SPhyREnvironment

env = SPhyREnvironment()
observation = env.reset(subject="full_hard")
scored = env.step(SPhyRAction(grid=completed_grid_text))
```

## `reset()` parameters

| Parameter | Meaning |
| --- | --- |
| `subject` | which of the 16 SPhyR subjects to draw from, e.g. `full_hard` |
| `sample_index` | serve a specific sample instead of the next one |
| `sample_count` | how many samples of the subject to serve (default 100) |
| `rotation_count` | quarter turns applied to the sample; the load case rotates with it |
| `few_shot_count` | solved examples to include in the prompt |
| `prompt_style` | `default`, `physics_enhanced`, `physics_neutral`, or your own template |
| `dataset_source` | `local` (bundled files) or `hub` (Hugging Face) |
| `seed` | reseed the sample order |

Arguments left out keep their current value, so a bare `reset()` serves the
next sample of the subject already loaded.

## Reward

`reward` is `topology_score`: stiffness relative to the best design achievable
on the reference material budget, discounted by any material spent beyond that
budget. Both halves are needed — stiffness alone is maximised by filling the
grid solid, thrift alone by building nothing. An unparsable or misshapen grid
scores `0.0`.

`observation.metrics` carries the benchmark's full metric set alongside it,
including `structural_efficiency`, `material_efficiency`, `compliance`,
`volume_ratio`, `load_carrying` and the string-matching metrics the benchmark
was originally scored with.

## Running the server

```bash
uv sync                     # from the repository root
cd envs/sphyr_env
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

The environment scores completions with the `sphyr` physics library, so the
image is built from the **repository root**:

```bash
docker build -f envs/sphyr_env/server/Dockerfile -t sphyr-env:latest .
docker run -p 8000:8000 sphyr-env:latest
```

The dataset files are baked into the image, so a container serves episodes
without network access.
