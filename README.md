# A-MoE-ASC


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