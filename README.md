# A-MoE-ASC


## SOTA


| Model                         | Rest14 Acc | Rest14 F1 | Lap14 Acc | Lap14 F1 | Rest15 Acc | Rest15 F1 | Rest16 Acc | Rest16 F1 | AVG F1 |
|------------------------------|------------|-----------|-----------|----------|------------|-----------|------------|-----------|--------|
| BERT-SPC                     | 84.11      | 76.68     | 77.59     | 73.28    | 83.48      | 66.18     | 90.10      | 74.16     | 72.58  |
| BERT-PT                      | 84.95      | 76.96     | 78.07     | 75.08    | –          | –         | –          | –         | –      |
| BERT-RGAT                    | 85.18      | 78.38     | 78.21     | 73.27    | 82.84      | 69.33     | 90.91      | 75.76     | 74.19  |
| BERT-DualGCN                 | 87.13      | 81.16     | 81.80     | 78.10    | –          | –         | –          | –         | –      |
| BERT-TGCN                    | 86.16      | 79.95     | 80.88     | 77.03    | 85.26      | 71.69     | 92.32      | 77.29     | 76.49  |
| BERT-SenticGCN               | 86.92      | 81.03     | 82.12     | 79.05    | 85.32      | 71.28     | 91.07      | 79.56     | 77.73  |
| **Llama 3 8B + Syn Chain**    | **91.87**  | **87.61** | **86.36** | **84.21**| **91.88**  | **83.65** | **95.94**  | **85.50** | **85.24** |



## Baseline

| Model                 | Dataset      | Fusion | Train loss | Val Acc | Test Acc |
| --------------------- | ------------ | ------ | ---------- | ------- | -------- |
| bert-base-uncased     | Laptop14     | Concat | 0.2208     | 0.7733  | 0.7461   |
| bert-base-uncased     | Laptop14     | Add    | 0.2131     | 0.7600  | 0.7492   |
| bert-base-uncased     | Laptop14     | Mul    | 0.2492     | 0.7533  | 0.7665   |
| bert-base-uncased     | Restaurant14 | Concat | 0.2444     | 0.7933  | 0.8071   |
| bert-base-uncased     | Restaurant14 | Add    | 0.2386     | 0.7867  | 0.8036   |
| bert-base-uncased     | Restaurant14 | Mul    | 0.2605     | 0.7467  | 0.8098   |
| roberta-base          | Laptop14     | Concat | 0.2623     | 0.7133  | 0.7947   |
| roberta-base          | Laptop14     | Add    | 0.2498     | 0.7400  | 0.7915   |
| roberta-base          | Laptop14     | Mul    | 0.2999     | 0.7533  | 0.7712   |
| roberta-base          | Restaurant14 | Concat | 0.2882     | 0.8000  | 0.8268   |
| roberta-base          | Restaurant14 | Add    | 0.2715     | 0.7667  | 0.8152   |
| roberta-base          | Restaurant14 | Mul    | 0.2826     | 0.8200  | 0.8223   |


## MoE


| Model                 | Dataset      | Fusion | Train loss | Val Acc | Test Acc |
| --------------------- | ------------ | ------ | ---------- | ------- | -------- |
| bert-base-uncased     | Laptop14     | Concat | 0.2208     | 0.7733  | 0.7461   |
| bert-base-uncased     | Laptop14     | Add    | 0.2131     | 0.7600  | 0.7492   |
| bert-base-uncased     | Laptop14     | Mul    | 0.2492     | 0.7533  | 0.7665   |
| bert-base-uncased     | Restaurant14 | Concat | 0.2444     | 0.7933  | 0.8071   |
| bert-base-uncased     | Restaurant14 | Add    | 0.2386     | 0.7867  | 0.8036   |
| bert-base-uncased     | Restaurant14 | Mul    | 0.2605     | 0.7467  | 0.8098   |
| roberta-base          | Laptop14     | Concat | 0.2623     | 0.7133  | 0.7947   |
| roberta-base          | Laptop14     | Add    | 0.2498     | 0.7400  | 0.7915   |
| roberta-base          | Laptop14     | Mul    | 0.2999     | 0.7533  | 0.7712   |
| roberta-base          | Restaurant14 | Concat | 0.2882     | 0.8000  | 0.8268   |
| roberta-base          | Restaurant14 | Add    | 0.2715     | 0.7667  | 0.8152   |
| roberta-base          | Restaurant14 | Mul    | 0.2826     | 0.8200  | 0.8223   |