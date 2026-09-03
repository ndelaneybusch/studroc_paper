suppressPackageStartupMessages(library(ROCnReg))
cat("ROCnReg version:", as.character(packageVersion("ROCnReg")), "\n")
print(args(pooledROC.BB))
d <- read.csv("data_t2_99.csv"); d1 <- d[d$rep == 0, ]
d1$group <- factor(ifelse(d1$label == 1, "D", "H"))
p <- c(0.998, 0.99, 0.98, 0.9, 0.5)
fit <- pooledROC.BB(marker = "score", group = "group", tag.h = "H", data = d1, p = p, B = 500, ci.level = 0.95)
cat("names:", names(fit), "\n"); str(fit$ROC); print(fit$p)
if (!is.null(fit$ROC)) print(cbind(p = fit$p, fit$ROC))
