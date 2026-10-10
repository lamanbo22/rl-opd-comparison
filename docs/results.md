# Results and metrics

## What the files contain

| File | Contents |
| :--- | :--- |
| `method_results.csv` | Correct counts and accuracies for every seed; method means and sample standard deviations |
| `paired_contrasts.csv` | All ten method comparisons and five comparisons with the initial student |
| `evaluation_results.csv` | Accuracy, invalid answers, truncation, and generation totals for all 46 evaluations |
| `training_costs.csv` | Actual training time, unused allowance, update counts, and phase shares for all 15 runs |
| `cost_breakdown.csv` | Component timings and recorded process attempts, including the failed evaluation attempt |
| `question_outcomes.csv` | Question-aligned correctness for every evaluated model |
| `analysis_manifest.json` | Source hashes, evaluation metadata, recorded costs, and the bootstrap specification |

All paths above are relative to `results/`.

## Question-level outcomes

The outcome file has one row per split and question: 64 halfway development rows, 512 final development rows, and 1,319 official test rows. `question_index` gives the zero-based order within a split; `source_row` identifies the row in the original GSM8K split. `question_sha256` hashes the normalized question text.

The remaining columns contain binary correctness for the initial student and each method/seed combination. Development baseline cells are empty because the 46-evaluation formal phase includes only the initial student's test evaluation. The same development questions appear in the halfway and full development sets.

Outcomes were exported from the completed evaluation records and checked against the finalized analysis: response-file hashes, summary hashes, ordered question identifiers, and correctness-vector hashes. The failed evaluation attempt contributes cost only; its completed replacement supplies the outcomes.

The released outcomes allow independent recomputation of accuracy and paired intervals. Invalid-answer counts, token totals, and timers are recorded metadata; recomputing those fields from raw responses requires the original run archive.

## Reading the metrics

- **Accuracy:** correct answers divided by all evaluated questions. The frozen parser accepts `FINAL_ANSWER:` or a `Final Answer:` fallback, then checks whether the parsed number matches the gold answer.
- **Mean accuracy:** arithmetic mean of the three seed accuracies. It describes the tested training configurations.
- **Seed SD:** sample standard deviation across the three training seeds, in percentage points.
- **Difference:** accuracy of `left` minus accuracy of `right`, in percentage points. For example, a change from 78% to 80% is +2 pp.
- **Paired interval:** a descriptive 95% bootstrap interval for that difference. An interval crossing zero leaves the direction of the difference unresolved by this analysis.
- **Invalid answers:** responses that fail the fixed answer-format or numerical parsing rules.
- **Truncated answers:** responses that reach the generation cap without the expected termination. These remain in the accuracy denominator and are scored by the same parser.

Bootstrap draws are paired across methods and independently resample training seeds and questions. The three seeds share an initial adapter. The 20,000 draws estimate uncertainty from these observations; they do not represent additional training runs.

## Reading the costs

`charged_seconds` measures work inside committed training batches. `session_wall_seconds` also includes setup and other session work. `updates` counts actual optimizer steps, while `rounds` includes batches with zero RL signal that skipped an update. Phase fractions divide charged time by signal type.

The component table uses `scope`, `name`, `method`, `seed`, `metric`, `seconds`, and `contained_in`. A component can be contained within another timer: generation is part of training core time, which is part of charged time. **Do not sum overlapping timer levels.** Empty timer values mean unavailable, not zero.

Every recorded attempt is retained. This includes the failed evaluation attempt, which is excluded from accuracy statistics. Training costs, evaluation costs, and total recorded process costs answer different resource questions.

## Recompute

```bash
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_results.py --check
```

This validates the outcome and pairing hashes, recomputes the statistics, compares all five tables, and writes regenerated tables to `artifacts/reproduced`. It also checks the bootstrap draw hash against the original analysis. Source results are never overwritten.
