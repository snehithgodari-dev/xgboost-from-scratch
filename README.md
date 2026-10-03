# XGBoost From Scratch: First Order vs Second Order Gradient Boosting

Implements classic gradient boosting (Friedman, 2001) and XGBoost-style boosting
(Chen & Guestrin, 2016) from scratch, to make concrete exactly what the XGBoost
paper adds: one equation  i.e a second-order Taylor expansion of the loss,
instead of using only the first-order gradient.

## The math

Vanilla gradient boosting fits a tree to the negative gradient only. XGBoost
minimizes a second-order Taylor expansion of the loss around the current
prediction:

```
L(y, y_hat + f(x)) ~= L(y, y_hat) + g*f(x) + 0.5*h*f(x)^2
```

This gives a closed-form optimal leaf weight, no line search needed:

```
w* = -G_j / (H_j + lambda)
```

`G_j`, `H_j` = summed gradients/Hessians of training points in leaf `j`.
`lambda` = L2 regularization term. The same expansion gives the exact-greedy
split-gain formula XGBoost uses to grow trees:

```
gain = 0.5 * [GL^2/(HL+lambda) + GR^2/(HR+lambda) - G^2/(H+lambda)] - gamma
```

`lambda` isn't a blind regularization knob , it directly discounts leaf
confidence when `H_j` (how much curvature/certainty the data gives in that
leaf) is small. That's why it matters most on small-sample leaves.

## What is implemented

1.`VanillaGBM`: first-order boosting. Fits a plain MSE regression tree
  (`sklearn.tree.DecisionTreeRegressor`) to the negative gradient. No Hessian
  used anywhere.
2.`XGBScratch`: second-order boosting. A from-scratch exact-greedy tree that
  splits on the gain formula above and sets leaf weights to `w* = -G/(H+lambda)`.
  Both share the identical boosting loop and logistic loss the only
  difference between the two classes is the split/leaf math.

## Verification

A controlled sanity check: run `XGBScratch` for a single boosting round
(`max_depth=2`, `learning_rate=1.0`, matched `lambda`/`gamma`) and compare
predicted probabilities against the real `xgboost` library with matching
hyperparameters.

**Result:**correlation `1.0000` between scratch and real `xgboost`
predictions; mean absolute difference `0.0089`. Not bit-identical, because
the real library defaults to `base_score=0.5` and uses float32 internals
even in `tree_method="exact"` mode but the near perfect correlation
confirms the gain and leaf weight formulas are implemented correctly.

## Data

Real Titanic passenger records (891 rows), from the public
[datasciencedojo/datasets](https://github.com/datasciencedojo/datasets)
CSV. Standard preprocessing: median-impute `Age`/`Fare`, mode-impute
`Embarked`, encode `Sex`/`Embarked`, engineer `FamilySize` and `IsAlone`.
80/20 stratified train/test split.

## Results (held-out 20% test set, 179 passengers)

| Model                                      | Accuracy |

| Single tree (depth=4) | 78.8% |
| Random Forest (100 trees) | 79.3% |
| Vanilla GBM from scratch (1st order) | 80.4% |
| XGBoost from scratch (2nd order, default) | 80.4% |
| XGBoost from scratch (2nd order, tuned via 4-fold CV grid search) | 80.4% |
| Real `xgboost` library (same tuned params) | 79.9% |

**Honest read:** the gap between first order and second order boosting is
small on this dataset. With only 179 test examples, one flipped prediction
moves accuracy by about 0.56 percentage points, so these differences are
close to measurement noise. XGBoost's real world edge over vanilla GBM shows
up more on larger, noisier, higher dimensional datasets where the Hessian
weighting and regularization actually have room to matter not
necessarily on a small, clean, 9 feature dataset like this one. Reporting
that honestly is more defensible than claiming a win that isn't really
there.

## SHAP feature analysis

Ranked features by mean `|SHAP value|` using `shap.TreeExplainer` on the
tuned real-`xgboost` model:

| Feature | Mean \|SHAP value\| |

| Sex | 1.326 |
| Pclass | 0.683 |
| Fare | 0.503 |
| Age | 0.419 |
| Embarked | 0.249 |
| FamilySize | 0.151 |
| SibSp | 0.043 |
| Parch | 0.026 |
| IsAlone | 0.000 |

`IsAlone` has essentially zero impact expected, since it's a deterministic
function of `FamilySize` (`FamilySize == 1`) and therefore redundant once
`FamilySize` is already a feature. Dropping it: **79.9% -> 79.9%**, no
accuracy cost. The useful result isn't an accuracy gain — it's a simpler
model with identical performance, found systematically via SHAP rather than
by guessing which features to drop.

## Repo structure

```
xgboost-from-scratch/
├── README.md
├── data_prep.ipynb                 # Titanic preprocessing
├── boosting_scratch.ipynb          # VanillaGBM + XGBScratch implementations
├── verify_and_benchmark.ipynb       # verification + full model comparison
├── shap_analysis.ipynb             # SHAP feature importance + feature selection test
├── titanic.csv                   # real dataset (891 passengers)
├── xgboost_explained.ipynb       # walkthrough with commentary
└── xgboost_code_only.ipynb       # clean code, no explanation
```

## How to run
open `.ipynb` in Jupyter/Colab and run all cells (takes a few
minutes, the hyperparameter grid search is the slow part).
