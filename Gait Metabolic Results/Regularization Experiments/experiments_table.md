| Population | Model | Settings | Best epoch | Test MAE (best) | Test MAE (final, ep 100) | Test r (best) | Test r (final) |
|---|---|---|---|---|---|---|---|
| Prosthetic | STGCN | Original (LR 0.01, no dropout) | 75 | 0.2162 | 0.2171 | 0.9102 | 0.9041 |
| Prosthetic | STGCN | LR 0.001 + dropout 0.5 | 88 | 0.2230 | 0.2260 | 0.9028 | 0.8999 |
| Prosthetic | STGCN | LR 0.001 + dropout 0.7 | 68 | 0.2432 | 0.2371 | 0.8914 | 0.8997 |
| Prosthetic | CTRGCN | Original (LR 0.01, no dropout) | 38 | 0.2321 | 0.2290 | 0.9037 | 0.8984 |
| Prosthetic | CTRGCN | LR 0.001 + dropout 0.5 | 88 | 0.2313 | 0.2419 | 0.8974 | 0.8930 |
| Prosthetic | CTRGCN | LR 0.001 + dropout 0.7 | 92 | 0.2401 | 0.2422 | 0.8973 | 0.8986 |
| Able-Bodied | STGCN | Original (LR 0.01, no dropout) | 33 | 0.3903 | 0.3913 | 0.9497 | 0.9476 |
| Able-Bodied | STGCN | LR 0.001 + dropout 0.5 | 34 | 0.3813 | 0.4205 | 0.9519 | 0.9414 |
| Able-Bodied | STGCN | LR 0.001 + dropout 0.7 | 55 | 0.3750 | 0.4440 | 0.9524 | 0.9267 |
| Able-Bodied | CTRGCN | Original (LR 0.01, no dropout) | 16 | 0.3756 | 0.3887 | 0.9539 | 0.9496 |
| Able-Bodied | CTRGCN | LR 0.001 + dropout 0.5 | 63 | 0.3865 | 0.3882 | 0.9509 | 0.9501 |
| Able-Bodied | CTRGCN | LR 0.001 + dropout 0.7 | 55 | 0.3757 | 0.3833 | 0.9543 | 0.9531 |
