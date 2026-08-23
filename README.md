# Classification Under Local Differential Privacy with Model Reversal and Model Averaging

Welcome! This repository is the code companion for our paper **"Classification Under Local Differential Privacy with Model Reversal and Model Averaging"**, by Caihong Qin and Yang Bai.

The paper was published in *Journal of Machine Learning Research*, 27(5):1--44, 2026. The JMLR paper page is: https://www.jmlr.org/papers/v27/24-0290.html.

The paper studies binary classification under local differential privacy, where each observation is perturbed before it is shared. The main idea is to train weak classifiers on privatized data, estimate their usefulness from privatized binary feedback, reverse weak classifiers that point in the wrong direction, and average useful classifiers.

In the training step, both the vector predictors `x` and labels `y` are privatized before the weak classifiers are fitted. In the evaluation step, the evaluation data are used only to release noisy binary feedback indicating whether each weak classifier is correct or incorrect.

## Example

The code below generates vector-valued sample data, applies the LDP mechanism, and reports held-out errors for the weak classifiers and the proposed aggregation methods.

Install `VGAM` if it is not already available:

```r
install.packages("VGAM")
```

```r
# Load the vector-valued LDP MRMA functions.
source("ldp_mrma_vector.R")

# Generate training, private-evaluation, and test datasets.
# Each dataset is a list(x, y), where x is a numeric matrix and y is coded -1/1.
# sample_data contains train, evaluation, test, and beta.
# Their x matrices have 4 columns and 500, 2500, and 500 rows, respectively.
# beta is the coefficient vector used to generate the labels.
sample_data <- generate_ldp_sample_data(seed = 2026)

train_data <- sample_data$train
evaluation_data <- sample_data$evaluation
test_data <- sample_data$test

# Fit the LDP classifier using privatized training data and privatized feedback.
fit <- fit_ldp_mrma_vector(
  train_data = train_data,                # data used to train weak classifiers
  evaluation_data = evaluation_data,      # data used to estimate weak-model utility
  epsilon = 10,                           # privacy budget
  learner = "logistic",                   # weak classifier type
  error_cutoff = 0.25,                    # cutoff for selecting weak classifiers
  seed = 2026                             # random seed for reproducibility
)

# Compare Weak, MR, MA, and MRMA on held-out test data.
method_errors <- evaluate_ldp_mrma_methods(fit, test_data)
method_errors
```

The reported values are held-out misclassification rates. The methods are:

- `Weak`: the average test error of the 50 individual weak classifiers trained on privatized data.
- `MR`: the average error after selecting useful weak classifiers and reversing those estimated to point in the wrong direction.
- `MA`: the error of the weighted average of useful weak classifiers without reversal.
- `MRMA`: the error after model reversal and weighted model averaging.

Running the example gives:

```text
  method  error
1   Weak 0.4674
2     MR 0.2324
3     MA 0.1820
4   MRMA 0.1700
```

The individual weak classifiers have an average misclassification rate of 46.74%. Model reversal reduces the average error to 23.24%, while model averaging gives 18.20%. Combining model reversal and model averaging performs best in this example, with a misclassification rate of 17.00%.

To use your own data, prepare `train_data`, `evaluation_data`, and `test_data` as lists with the form `list(x = x_matrix, y = y_vector)`. The matrix `x` should contain numeric vector-valued predictors scaled to the bounded range used by the LDP mechanism, here `[-1, 1]`, and `y` should be coded as `-1` and `1`. The code assumes this structure directly and does not recode labels, reshape data, or perform preprocessing.

## Details

The repository includes:

- `ldp_mrma_vector.R`: vector-valued data generation, LDP perturbation, weak classifier fitting, privatized feedback correction, model reversal, and model averaging.

Required R packages:

- `VGAM` is used for Laplace noise generation in the LDP perturbation.
- `stats` is used from standard R for logistic weak classifiers.
- `e1071` is only needed if you set `learner = "svm"`.

Main functions:

- `generate_ldp_sample_data()`: generates training, private-evaluation, and test datasets for a small vector-valued example.
- `fit_ldp_mrma_vector()`: fits weak classifiers and combines them using MR, MA, and MRMA.
- `evaluate_ldp_mrma_methods()`: reports test errors for Weak, MR, MA, and MRMA.
