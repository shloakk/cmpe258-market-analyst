# Multi-Model Evaluation Report

Generated from eval run: `20260507T102711Z`

## Model Comparison

| Model | F1 | Precision | Recall | Hallucination | Theme Coverage | Theme Purity | Cost | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt | 0.633 | 0.518 | 0.900 | 0.000 | 0.333 | 0.142 | $0.178568 | 12.1s |
| llama | 0.491 | 0.382 | 0.847 | 0.000 | 0.408 | 0.145 | $0.017082 | 7.4s |

## Source Result Files

- `gpt`: `/Users/matthewbernard/Documents/GitHub/cmpe258-market-analyst/eval/results/gpt_20260507T102355Z.json`
- `llama`: `/Users/matthewbernard/Documents/GitHub/cmpe258-market-analyst/eval/results/llama_20260507T102556Z.json`

## Notes

- `Hallucination` is lower-is-better.
- `Theme Purity` is corpus-tag homogeneity inside predicted themes; higher is better.
- Cost includes Mapper/Critic LLM calls and any JSON retry attempts.
