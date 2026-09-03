suppressPackageStartupMessages(library(GET))
cat("GET version:", as.character(packageVersion("GET")), "\n")
cloud <- as.matrix(read.csv("cloud.csv", header = FALSE))      # M x K
ours <- read.csv("ours.csv")
r <- ours$t; K <- length(r); M <- nrow(cloud)
cs <- create_curve_set(list(r = r, obs = t(cloud)))            # obs: K x M
for (type in c("rank", "erl")) {
  cr <- central_region(cs, type = type, coverage = 0.95, alternative = "two.sided")
  lo <- cr$lo; hi <- cr$hi
  cat(sprintf("\ntype = %s\n", type))
  cat(sprintf("  max |lo_GET - lo_ours| = %.3e ; max |hi_GET - hi_ours| = %.3e\n", max(abs(lo - ours$lo)), max(abs(hi - ours$hi))))
  cat(sprintf("  grid points where GET is strictly wider than ours: %d ; strictly narrower: %d (of %d)\n",
              sum(lo < ours$lo - 1e-12 | hi > ours$hi + 1e-12), sum(lo > ours$lo + 1e-12 | hi < ours$hi - 1e-12), K))
  cat(sprintf("  truth inside GET region at all grid points: %s ; inside ours: %s\n",
              all(ours$truth >= lo - 1e-12 & ours$truth <= hi + 1e-12), all(ours$truth >= ours$lo - 1e-12 & ours$truth <= ours$hi + 1e-12)))
  viol <- which(ours$truth < lo - 1e-12 | ours$truth > hi + 1e-12)
  if (length(viol)) cat("  GET violations at t =", head(round(r[viol], 3), 12), if (length(viol) > 12) "..." else "", "\n")
  if (type == "rank") {
    # GET's retained set vs ours: curves inside the k-th envelope
    inside <- colSums(t(cloud) >= lo - 1e-12 & t(cloud) <= hi + 1e-12) == K
    cat(sprintf("  fraction of cloud curves inside GET rank region: %.4f\n", mean(inside)))
  }
}
