# ROCnReg pooledROC.BB pointwise credible band: frequentist coverage at fixed FPR grid points
# inside the hook region, on the wedge cell t(2)/.99 (n=500/500), the sliver DGP (AUC .80), and a
# concave-corner reference t(30)/.95. Truth at each p from truth_<shape>.csv.
suppressPackageStartupMessages(library(ROCnReg))
args <- commandArgs(trailingOnly = TRUE)
B <- if (length(args) >= 1) as.integer(args[1]) else 2000
nrep <- if (length(args) >= 2) as.integer(args[2]) else 200
for (shape in c("t2_99", "sliver80", "t30_95")) {
  d <- read.csv(sprintf("data_%s.csv", shape)); tr <- read.csv(sprintf("truth_%s.csv", shape))
  p <- tr$p; K <- length(p)
  miss_lo <- numeric(K); miss_hi <- numeric(K); degenerate <- numeric(K); width <- numeric(K); done <- 0
  for (r in 0:(nrep - 1)) {
    dr <- d[d$rep == r, ]; dr$group <- factor(ifelse(dr$label == 1, "D", "H"))
    fit <- tryCatch(pooledROC.BB(marker = "score", group = "group", tag.h = "H", data = dr, p = p, B = B, ci.level = 0.95),
                    error = function(e) { cat("rep", r, "error:", conditionMessage(e), "\n"); NULL })
    if (is.null(fit)) next
    roc <- fit$ROC; ql <- roc[, "ql"]; qh <- roc[, "qh"]
    miss_lo <- miss_lo + (tr$truth < ql - 1e-12); miss_hi <- miss_hi + (tr$truth > qh + 1e-12)
    degenerate <- degenerate + (qh - ql < 1e-12); width <- width + (qh - ql); done <- done + 1
  }
  cat(sprintf("\n%s: ROCnReg pooledROC.BB pointwise 95%% band, B=%d, %d reps\n", shape, B, done))
  cat(sprintf("  %4s %7s %10s %9s %9s %9s %9s %9s\n", "k", "FPR", "truth", "low-miss", "up-miss", "coverage", "P(degen)", "mean wid"))
  for (i in seq_len(K)) cat(sprintf("  %4d %7.3f %10.6f %9.3f %9.3f %9.3f %9.3f %9.5f\n", tr$k[i], p[i], tr$truth[i],
                                    miss_lo[i] / done, miss_hi[i] / done, 1 - (miss_lo[i] + miss_hi[i]) / done, degenerate[i] / done, width[i] / done))
  write.csv(data.frame(k = tr$k, p = p, truth = tr$truth, low_miss = miss_lo / done, up_miss = miss_hi / done,
                       coverage = 1 - (miss_lo + miss_hi) / done, p_degenerate = degenerate / done, mean_width = width / done),
            sprintf("bb_coverage_%s.csv", shape), row.names = FALSE)
}
