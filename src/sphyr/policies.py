"""Policies that act in the SPhyR environment.

A policy maps a :class:`sphyr_env.SPhyRObservation` to a
:class:`sphyr_env.SPhyRAction`.  The benchmark's own policies are language
models: they are shown ``observation.prompt`` and their reply is submitted as
the completed grid.

Every model is reached through OpenRouter, which speaks the OpenAI chat
completions API for all of them.  One endpoint, one key, one call style: the
benchmark no longer carries a provider SDK and a call style per vendor, and
adding a model is a line in :data:`MODELS`.

The OpenAI SDK is imported inside the call, not at module scope: rescoring
stored results and running the environment itself do not need it.
"""

import os
import random
import time
from typing import Optional

from dotenv import load_dotenv

from sphyr.tasks import RANDOM_SEED, grid_to_str, str_to_grid
from sphyr_env import SPhyRAction

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

# Models served by a local Ollama daemon, addressed as ``ollama/<model>``.
# Ollama speaks the same chat-completions API, so a local model is reached the
# same way as a hosted one and needs no key.
OLLAMA_PREFIX = "ollama/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Enough headroom for a completed grid plus whatever a reasoning model thinks
# out loud on its way there; the grid is parsed back out of the reply.
#
# Reasoning tokens count against this budget, and they dominate it: one
# DeepSeek-R1 sample on ``full_easy`` spent 4,907 reasoning tokens to produce a
# 199-character grid, for 6,717 completion tokens in total.  A 4,096 cap
# truncates that mid-thought and returns no grid at all, which scores as a
# model failure rather than the configuration error it is.  Unused headroom is
# free -- only generated tokens are billed -- so the cap is set well above the
# worst case observed rather than close to it.
MAX_TOKENS = 16000

# The benchmarked models, keyed by the name the published results are stored
# under (``results/<model>/``) and mapped to the OpenRouter model they are run
# through.  Keeping the result names stable is what lets a fresh run be
# compared against the numbers in the paper.
#
# A ``None`` marks a model OpenRouter no longer serves.  The stored results for
# it stay readable and re-scorable; it simply cannot be run again, and saying
# so here beats a 404 from the API halfway through an experiment.
MODELS: dict[str, Optional[str]] = {
    "gpt-4.1-2025-04-14": "openai/gpt-4.1",
    "gpt-4o-2024-08-06": "openai/gpt-4o-2024-08-06",
    # OpenRouter serves only the moving alias, which currently resolves to the
    # 0125 snapshot the paper reported.
    "gpt-3.5-turbo-0125": "openai/gpt-3.5-turbo",
    "gemini-1.5-pro": None,
    "gemini-2.5-pro-preview-05-06": "google/gemini-2.5-pro-preview-05-06",
    "claude-3-7-sonnet-20250219": None,
    "claude-opus-4-20250514": "anthropic/claude-opus-4",
    # The paper's "DeepSeek-R1" column was produced against deepseek-chat, so
    # that is what it maps to here.
    "deepseek-reasoner": "deepseek/deepseek-chat",
    "perplexity-sonar": "perplexity/sonar",
    "perplexity-sonar-reasoning": None,
}

RUNNABLE_MODELS = sorted(name for name, slug in MODELS.items() if slug)


def resolve_model(name: str) -> str:
    """Map a benchmark model name to the OpenRouter model to call.

    Any OpenRouter model can be benchmarked by naming it directly
    (``anthropic/claude-sonnet-4.5``); the names in :data:`MODELS` are the ones
    the published results were produced with.
    """
    if name in MODELS:
        slug = MODELS[name]
        if slug is None:
            raise ValueError(
                f"{name} is no longer served by OpenRouter, so it cannot be run "
                "again; its stored results can still be rescored. Runnable "
                f"benchmark models: {', '.join(RUNNABLE_MODELS)}"
            )
        return slug

    if name.startswith(OLLAMA_PREFIX):
        # Local model: everything after the prefix is the Ollama model tag.
        return name[len(OLLAMA_PREFIX) :]

    if "/" in name:
        # An OpenRouter model id, passed through as given.
        return name

    raise ValueError(
        f"Unknown model: {name}; expected a benchmark model "
        f"({', '.join(RUNNABLE_MODELS)}) or an OpenRouter model id like "
        "'anthropic/claude-sonnet-4.5'"
    )


# A reasoning model can take minutes on one grid -- the slowest calls observed
# ran past eight -- so the client waits far longer than the SDK's default
# before giving up on a reply that is still coming.
REQUEST_TIMEOUT = 1800

# Transient failures are the norm at this call volume rather than the
# exception: over a full 1,600-sample run, 2.3% of calls failed and every one
# of them succeeded on a later attempt.  Three observed modes, all retryable:
# an empty reply, a response body that stops being valid JSON partway, and a
# request that times out.
MAX_ATTEMPTS = 5
RETRY_BACKOFF = 4


def _client(name=""):
    from openai import OpenAI

    if name.startswith(OLLAMA_PREFIX):
        # The local daemon ignores the key but the SDK insists on one.
        return OpenAI(
            api_key="ollama", base_url=OLLAMA_BASE_URL, timeout=REQUEST_TIMEOUT
        )

    key = os.getenv(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set; add it to your .env "
            "(get one at https://openrouter.ai/keys)"
        )

    return OpenAI(api_key=key, base_url=BASE_URL, timeout=REQUEST_TIMEOUT)


def call_model(
    name: str,
    prompt: str,
    max_tokens: int = MAX_TOKENS,
    max_attempts: int = MAX_ATTEMPTS,
) -> str:
    """Send one prompt to a benchmarked model through OpenRouter.

    Retries transient failures with exponential backoff.  Without this a failed
    call drops the sample from the run, and because a call fails most readily
    where the model reasons longest, the samples lost are the hardest ones --
    which biases the surviving scores upward.
    """
    model = resolve_model(name)

    # Built once, outside the loop: a missing key is a configuration error, not
    # a transient one, and retrying it only delays saying so.
    client = _client(name)
    last_error = None

    for attempt in range(max_attempts):
        if attempt:
            time.sleep(RETRY_BACKOFF * 2 ** (attempt - 1))

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            if not content:
                raise RuntimeError(f"{model} returned an empty reply")

            return content.strip()
        except Exception as error:  # noqa: BLE001 - every mode seen is transient
            last_error = error

    raise RuntimeError(
        f"{model} failed {max_attempts} times; last error: {last_error}"
    ) from last_error


class LLMPolicy:
    """Ask a language model to complete the grid it is shown."""

    def __init__(self, model: str, max_tokens: int = MAX_TOKENS):
        # Fail here rather than on the first API call, so a typo in a model
        # name does not surface an hour into an experiment.
        resolve_model(model)
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, observation) -> SPhyRAction:
        return SPhyRAction(
            grid=call_model(self.model, observation.prompt, self.max_tokens)
        )


class ConstantFillPolicy:
    """Fill every masked cell with one value.

    A baseline rather than a contender: filling solid maximises stiffness and
    fails on material, filling empty does the reverse.  Useful for checking
    that the environment and its reward are wired up without spending API
    calls.
    """

    def __init__(self, value: str = "1"):
        self.value = value

    def __call__(self, observation) -> SPhyRAction:
        grid = [
            [self.value if cell == "V" else cell for cell in row]
            for row in str_to_grid(observation.grid)
        ]
        return SPhyRAction(grid="\n".join(" ".join(row) for row in grid))


class RandomFillPolicy:
    """Fill every masked cell with a random admissible value.

    The floor a model has to clear.  On binary subjects each masked cell is
    0 or 1 with equal probability; on density subjects it is a value in
    [0, 1] to one decimal place, matching what those subjects accept.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self._rng = random.Random(seed)

    def __call__(self, observation) -> SPhyRAction:
        density = observation.cell_values == "density"
        grid = [
            [
                (
                    f"{self._rng.randint(0, 10) / 10:.1f}"
                    if density
                    else str(self._rng.randint(0, 1))
                )
                if cell == "V"
                else cell
                for cell in row
            ]
            for row in str_to_grid(observation.grid)
        ]
        return SPhyRAction(grid="\n".join(" ".join(row) for row in grid))


class RetrievalPolicy:
    """Answer with the solution to the most similar sample in the dataset.

    The question this baseline settles is how much of SPhyR can be solved
    without reasoning at all.  For each query it scores every record of the
    subject by agreement on the cells the query leaves *visible*, then copies
    the best match's answer into the masked cells.

    The query's own record is excluded from the corpus, so a hit is a genuine
    nearest neighbour rather than the sample retrieving itself.  A high score
    here would mean the benchmark rewards memorising the dataset's load cases;
    a low one means the masked cells are not recoverable from a similar
    boundary condition and have to be reasoned about.
    """

    def __init__(self, dataset_source=None, exclude_visible_matches: bool = True):
        self._corpus = {}
        self._dataset_source = dataset_source
        # A record that agrees with the query on every visible cell is, for
        # this dataset, the query's own structure under a different mask --
        # retrieving it is looking up the answer rather than generalising from
        # a neighbour.  Excluded by default; the unexcluded variant is reported
        # separately, because how far it rises is a fact about the dataset.
        self._exclude_visible_matches = exclude_visible_matches

    def _records(self, subject):
        if subject not in self._corpus:
            from sphyr.tasks import load_subject

            self._corpus[subject] = [
                (str_to_grid(grid_to_str(r["input_grid"])),
                 str_to_grid(grid_to_str(r["ground_truth"])))
                for r in load_subject(subject, source=self._dataset_source)
            ]
        return self._corpus[subject]

    def __call__(self, observation) -> SPhyRAction:
        query = str_to_grid(observation.grid)
        visible = [
            (i, j)
            for i, row in enumerate(query)
            for j, cell in enumerate(row)
            if cell != "V"
        ]
        masked = [
            (i, j)
            for i, row in enumerate(query)
            for j, cell in enumerate(row)
            if cell == "V"
        ]

        best, best_score = None, -1
        for candidate_input, candidate_truth in self._records(observation.subject):
            if len(candidate_truth) != len(query):
                continue
            score = sum(
                1
                for i, j in visible
                if j < len(candidate_truth[i]) and candidate_truth[i][j] == query[i][j]
            )
            # Perfect agreement on everything visible means the same structure
            # under a different mask, so the "retrieval" is a lookup.
            if self._exclude_visible_matches and score == len(visible):
                continue
            if score > best_score:
                best, best_score = candidate_truth, score

        if best is None:
            return SPhyRAction(grid=observation.grid)

        filled = [list(row) for row in query]
        for i, j in masked:
            if i < len(best) and j < len(best[i]):
                filled[i][j] = best[i][j]
        return SPhyRAction(grid="\n".join(" ".join(row) for row in filled))


# Baselines that answer without a model, addressed by name like any other
# policy so that a baseline run and a model run are measured identically.
BASELINES = {
    "baseline-solid": lambda: ConstantFillPolicy("1"),
    "baseline-empty": lambda: ConstantFillPolicy("0"),
    "baseline-random": RandomFillPolicy,
    "baseline-retrieval": RetrievalPolicy,
    # Same policy with the lookup left in, to quantify how much of the
    # benchmark is answerable by finding the same structure elsewhere.
    "baseline-retrieval-lookup": lambda: RetrievalPolicy(
        exclude_visible_matches=False
    ),
}


def get_policy(model: str):
    """Build the policy for a benchmarked model name, or a named baseline."""
    if model in BASELINES:
        return BASELINES[model]()
    return LLMPolicy(model)
