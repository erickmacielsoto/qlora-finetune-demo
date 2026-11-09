| Model | n | Accuracy | 95% CI (Wilson) | Parse fail | Hit token limit | p50 s/item | p95 s/item | Mean out tokens | Tok/s | Wall time s | GPU cost USD | USD / 1k correct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hf:outputs/base-chat` | 300 | 28.0% | 23.2% to 33.3% | 0 | 178 | 0.942 | 1.227 | 297 | 308 | 289.7 | 0.0161 | 0.1916 |
| `hf:outputs/merged` | 300 | 30.0% | 25.1% to 35.4% | 0 | 240 | 0.944 | 1.036 | 328 | 342 | 287.6 | 0.0160 | 0.1775 |

GPU cost assumes USD 0.20 per GPU-hour applied to wall time.

| A | B | Acc(A) - Acc(B) | 95% CI (paired bootstrap) | Only A correct | Only B correct | McNemar exact p | Significant (alpha=0.05) |
|---|---|---:|---|---:|---:|---:|---|
| `hf:outputs/base-chat` | `hf:outputs/merged` | -2.0 pp | -8.3 to +4.3 pp | 45 | 51 | 0.61 | no |
