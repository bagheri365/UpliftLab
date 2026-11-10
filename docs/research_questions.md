# Research questions and hypotheses

## Primary question

**Can a policy selecting customers by estimated incremental purchase response outperform treating every customer or treating no customers?**

A causal interpretation requires defensible treatment assignment and pre-treatment covariates. Otherwise we can compare predictive targeting scores but should not claim causal lift.

## Secondary questions

1. How do S-, T-, X-, R-, and DR-Learners and Causal Forest compare on a shared feature set and split?
2. How stable are rankings and estimated policy values across random seeds and bootstrap samples?
3. Does historical purchase activity help identify heterogeneous treatment response beyond demographics?
4. How do computation time and complexity compare?
5. On ACIC semi-synthetic benchmarks, how accurately do models estimate known treatment effects?

## Formal estimand

For pre-treatment covariates X, treatment T in {0,1}, and binary purchase outcome Y:

    tau(x) = E[Y(1) - Y(0) | X=x]

A policy pi(X) chooses treatment if estimated incremental value exceeds its relevant cost or threshold. Individual counterfactual outcomes cannot both be observed for the same customer.

## Falsifiable hypotheses

- H1: A learned policy improves *estimated* value over treat-all and treat-none on held-out data.
- H2: Behavioral features improve ranking relative to demographic-only features.
- H3: Relative model performance varies across policy budgets and outcome distributions.

No hypotheses are marked supported until evaluation is complete.
