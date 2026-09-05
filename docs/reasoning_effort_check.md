# Does `reasoning_effort` explain the model gap?

Both `anthropic/claude-opus-5` and `google/gemini-3.8-flash` advertise
`reasoning_effort`, and the two were benchmarked at their defaults. At those
defaults Gemini generates several times more tokens than Opus on identical
prompts, so any score difference between them confounds how much each model
deliberates with how well it reasons. This checks whether the parameter
accounts for the gap.

`full_easy` on `v2-20`, 25 samples per condition, scored through the
environment:

| model | effort | output tokens | Topology Score | Load Carrying |
| --- | --- | --- | --- | --- |
| claude-opus-5 | default | 3,170 | 7.67 | 48.0 |
| claude-opus-5 | high | 3,242 | 6.50 | 40.0 |
| gemini-3.8-flash | default | 4,582 | 21.25 | 88.0 |
| gemini-3.8-flash | high | 15,964 | 0.00 | 0.0 |

## Conclusions

**The parameter is inert for Opus.** Output grows 2% and the score does not
improve. Opus's lower score is therefore not an artefact of under-thinking that
the parameter can correct, and at the highest effort available it still scores
6.50 against Gemini's 21.25. The comparison survives as a capability claim.

**Raising effort can destroy a score outright.** Gemini at high effort emits
15,964 tokens against a 16,000 cap and returns no grid at all on any of the 25
samples, scoring 0.00. It reasons until the budget is exhausted. This is the
same failure DeepSeek-R1 showed against a 4,096 cap, at a higher ceiling, and
it is worth stating as a practical hazard: on tasks whose answers are long,
increasing reasoning effort can reduce a benchmark score to zero without
producing a single malformed answer to diagnose it from.

## A note on sample size

An earlier version of this check used five samples and suggested Opus gained
11.6 Topology Score at high effort. At twenty-five the effect is -1.2. The
five-sample result was noise, and the same subject varies by twenty points
between a five-sample and a hundred-sample estimate. Effects of this size
cannot be measured at that scale.
