# Compute the privacy quantities used by the vector LDP mechanism.
make_ldp_privacy <- function(epsilon, p) {
  epsilon_xy <- epsilon / (p + 1)
  list(
    epsilon = epsilon,
    q_label = stats::plogis(epsilon_xy),
    q_feedback = stats::plogis(epsilon),
    x_noise_scale = 2 * (p + 1) / epsilon
  )
}

# Perturb vector features and labels under local differential privacy.
perturb_ldp_data <- function(x, y, privacy) {
  if (!requireNamespace("VGAM", quietly = TRUE)) {
    stop("Install package 'VGAM' to generate Laplace noise.", call. = FALSE)
  }
  y_private <- y * (2 * stats::rbinom(length(y), size = 1, prob = privacy$q_label) - 1)
  x_private <- x + matrix(
    VGAM::rlaplace(length(x), location = 0, scale = privacy$x_noise_scale),
    nrow = nrow(x)
  )
  list(x = x_private, y = y_private)
}

# Fit one weak linear classifier on privatized data.
fit_weak_classifier <- function(x, y, learner = c("logistic", "svm")) {
  learner <- match.arg(learner)

  if (learner == "logistic") {
    data <- as.data.frame(x)
    data$y <- ifelse(y == 1, 1, 0)
    model <- suppressWarnings(stats::glm(y ~ ., data = data, family = stats::binomial()))
    coefficients <- stats::coef(model)
    return(as.numeric(coefficients))
  }

  if (!requireNamespace("e1071", quietly = TRUE)) {
    stop("Install package 'e1071' to use learner = 'svm'.", call. = FALSE)
  }
  model <- e1071::svm(
    x = x,
    y = factor(y, levels = c(-1, 1)),
    kernel = "linear",
    type = "C-classification",
    probability = TRUE
  )
  as.numeric(stats::coef(model))
}

# Compute decision scores for a coefficient vector or coefficient matrix.
linear_scores <- function(coefficients, x) {
  if (!is.matrix(coefficients)) {
    coefficients <- matrix(coefficients, ncol = 1)
  }
  sweep(x %*% coefficients[-1, , drop = FALSE], 2, coefficients[1, ], "+")
}

# Convert decision scores to labels in {-1, 1}.
linear_labels <- function(coefficients, x) {
  ifelse(linear_scores(coefficients, x) >= 0, 1, -1)
}

# Compute the misclassification rate for each coefficient vector.
classification_error <- function(coefficients, x, y) {
  predictions <- linear_labels(coefficients, x)
  colMeans(predictions != y)
}

# Estimate weak-classifier error using privatized binary feedback.
estimate_feedback_error <- function(x_eval, y_eval, coefficients, q_feedback) {
  true_error <- as.numeric(classification_error(coefficients, x_eval, y_eval))
  error_matrix <- linear_labels(coefficients, x_eval) != y_eval
  keep_original <- matrix(
    stats::rbinom(length(error_matrix), size = 1, prob = q_feedback),
    nrow = nrow(error_matrix)
  )
  private_error_matrix <- error_matrix
  private_error_matrix[keep_original == 0] <- !private_error_matrix[keep_original == 0]
  observed_error <- colMeans(private_error_matrix)
  corrected_error <- (observed_error + q_feedback - 1) / (2 * q_feedback - 1)
  corrected_error <- pmin(1, pmax(0, corrected_error))

  list(
    true = true_error,
    observed = observed_error,
    corrected = corrected_error
  )
}

# Average weak classifiers, with optional model reversal before averaging.
average_weak_classifiers <- function(coefficients, estimated_errors, error_cutoff = 0.20, reverse = TRUE) {
  utility <- 0.5 - estimated_errors
  threshold <- 0.5 - error_cutoff
  selected <- if (reverse) {
    which(abs(utility) > threshold)
  } else {
    which(utility > threshold)
  }
  if (length(selected) == 0) {
    return(list(
      coefficients = rep(NA_real_, nrow(coefficients)),
      selected = integer(0),
      weights = numeric(0),
      utility = numeric(0)
    ))
  }

  selected_coefficients <- coefficients[, selected, drop = FALSE]
  if (reverse) {
    selected_coefficients <- sweep(selected_coefficients, 2, sign(utility[selected]), "*")
  }
  weights <- abs(utility[selected])
  weights <- weights / sum(weights)

  list(
    coefficients = as.numeric(selected_coefficients %*% weights),
    selected = selected,
    weights = weights,
    utility = utility[selected]
  )
}

# Reverse weak classifiers without averaging them.
reverse_weak_classifiers <- function(coefficients, estimated_errors, error_cutoff = 0.20) {
  utility <- 0.5 - estimated_errors
  threshold <- 0.5 - error_cutoff
  selected <- which(abs(utility) > threshold)
  if (length(selected) == 0) {
    return(list(
      coefficients = matrix(NA_real_, nrow = nrow(coefficients), ncol = 1),
      selected = integer(0),
      utility = numeric(0)
    ))
  }
  signs <- sign(utility[selected])
  signs[signs == 0] <- 1

  list(
    coefficients = sweep(coefficients[, selected, drop = FALSE], 2, signs, "*"),
    selected = selected,
    utility = utility[selected]
  )
}

# Generate sample vector data for the README example.
generate_ldp_sample_data <- function(
  n_train = 500,
  n_eval = 2500,
  n_test = 500,
  p = 4,
  signal_strength = 5,
  seed = NULL
) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  n <- n_train + n_eval + n_test
  x <- matrix(stats::runif(n * p, min = -1, max = 1), ncol = p)
  beta <- (-1)^(seq_len(p) + 1) / seq_len(p)
  score <- as.vector(signal_strength * x %*% beta)
  y <- ifelse(stats::rbinom(n, 1, stats::plogis(score)) == 1, 1, -1)

  train_index <- seq_len(n_train)
  eval_index <- n_train + seq_len(n_eval)
  test_index <- n_train + n_eval + seq_len(n_test)

  list(
    train = list(x = x[train_index, , drop = FALSE], y = y[train_index]),
    evaluation = list(x = x[eval_index, , drop = FALSE], y = y[eval_index]),
    test = list(x = x[test_index, , drop = FALSE], y = y[test_index]),
    beta = beta
  )
}

# Fit weak classifiers and combine them using MR, MA, and MRMA.
fit_ldp_mrma_vector <- function(
  train_data,
  evaluation_data,
  epsilon = 5,
  learner = c("logistic", "svm"),
  n_weak = 50,
  weak_sample_size = 50,
  error_cutoff = 0.20,
  seed = NULL
) {
  learner <- match.arg(learner)
  if (!is.null(seed)) {
    set.seed(seed)
  }

  x_train <- train_data$x
  y_train <- train_data$y
  x_eval <- evaluation_data$x
  y_eval <- evaluation_data$y

  privacy <- make_ldp_privacy(epsilon, ncol(x_train))
  private_train <- perturb_ldp_data(x_train, y_train, privacy)

  eval_order <- sample(seq_len(nrow(x_eval)))
  eval_groups <- split(eval_order, rep(seq_len(n_weak), length.out = nrow(x_eval)))

  coefficients <- matrix(NA_real_, nrow = ncol(x_train) + 1, ncol = n_weak)
  true_errors <- observed_errors <- corrected_errors <- numeric(n_weak)

  for (j in seq_len(n_weak)) {
    train_index <- sample(seq_len(nrow(private_train$x)), size = weak_sample_size, replace = FALSE)
    coefficients[, j] <- fit_weak_classifier(
      x = private_train$x[train_index, , drop = FALSE],
      y = private_train$y[train_index],
      learner = learner
    )

    feedback <- estimate_feedback_error(
      x_eval = x_eval[eval_groups[[j]], , drop = FALSE],
      y_eval = y_eval[eval_groups[[j]]],
      coefficients = coefficients[, j],
      q_feedback = privacy$q_feedback
    )
    true_errors[j] <- feedback$true
    observed_errors[j] <- feedback$observed
    corrected_errors[j] <- feedback$corrected
  }

  mr <- reverse_weak_classifiers(coefficients, corrected_errors, error_cutoff = error_cutoff)
  ma <- average_weak_classifiers(coefficients, corrected_errors, error_cutoff = error_cutoff, reverse = FALSE)
  mrma <- average_weak_classifiers(coefficients, corrected_errors, error_cutoff = error_cutoff, reverse = TRUE)

  structure(
    list(
      learner = learner,
      weak_coefficients = coefficients,
      corrected_errors = corrected_errors,
      observed_errors = observed_errors,
      true_errors = true_errors,
      mr = mr,
      ma = ma,
      mrma = mrma,
      privacy = privacy,
      error_cutoff = error_cutoff
    ),
    class = "ldp_mrma_vector"
  )
}

# Evaluate Weak, MR, MA, and MRMA classifiers on test data.
evaluate_ldp_mrma_methods <- function(fit, test_data) {
  weak_error <- mean(classification_error(fit$weak_coefficients, test_data$x, test_data$y))
  mr_error <- mean(classification_error(fit$mr$coefficients, test_data$x, test_data$y))
  ma_error <- classification_error(fit$ma$coefficients, test_data$x, test_data$y)
  mrma_error <- classification_error(fit$mrma$coefficients, test_data$x, test_data$y)

  data.frame(
    method = c("Weak", "MR", "MA", "MRMA"),
    error = as.numeric(c(weak_error, mr_error, ma_error, mrma_error))
  )
}
