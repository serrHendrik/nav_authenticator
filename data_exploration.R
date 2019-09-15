
setwd('D:/Documents/SCHOOL/Ingenieurswetenschappen - Computerwetenschappen/KUL/Jaar II/Industriele Stage - ESA/dev/nav_authenticator')
#setwd('C:/Users/nsr/Documents/Hendrik/dev/nav_authenticator')


#filename <- "nav_data/10-09-2019_toe7200.csv"
filename <- "nav_data/10-09-2019_toe1800.csv"
data_orig <- read.csv(filename, header = TRUE)
sample_ind <- seq(0,length(data_orig$svId),60)
data <- data_orig[sample_ind,]


svId <- 9
ids <- data[,"svId"] == svId
X <- data[ids,]

plot(X$sp_X)
plot(X$sp_Y)
plot(X$sp_Z)
plot(X$sv_X)
plot(X$sv_X^2)
plot(X$sv_Y)
plot(X$sv_Z)
plot(X$svCb, xlab = "Time [s]", ylab = "delta Clock bias", main = paste("Satellite Clock Bias offset [SV E", svId,"]", sep = ""))

svIds <- c(1:36)
index_offset_svId <- rep(0,36)
for (i in svIds) {
  indices <- data[,"svId"] == i
  index_offset_svId[i] <- nrow(data[indices,])
}
index_offset_svId <- cumsum(index_offset_svId)

################################# Plots for all sats #############################
#sp_X
plot(data$sp_X, xlab = "", xaxt='n', ylab = "sp_X [m]", main = "Satellite Position offset sp_X")
for (i in index_offset_svId) {
  abline(v = i)
}
#sp_Y
plot(data$sp_Y, xlab = "", xaxt='n', ylab = "sp_Y [m]", main = "Satellite Position offset sp_Y [E01-E36]")
for (i in index_offset_svId) {
  abline(v = i)
}
#sp_Z
plot(data$sp_Z, xlab = "", xaxt='n', ylab = "sp_Z [m]", main = "Satellite Position offset sp_Z [E01-E36]")
for (i in index_offset_svId) {
  abline(v = i)
}

#Plot sv_X for all satellites
plot(data$sv_X, xlab = "", xaxt='n', ylab = "sv_X", main = "Satellite Velocity offset sv_X [E01-E36]")
for (i in index_offset_svId) {
  abline(v = i)
}
#Plot sv_Y for all satellites
plot(data$sv_Y, xlab = "", xaxt='n', ylab = "sv_Y", main = "Satellite Velocity offset sv_Y [E01-E36]")
for (i in index_offset_svId) {
  abline(v = i)
}
#Plot sv_Z for all satellites
plot(data$sv_Z, xlab = "", xaxt='n', ylab = "sv_Z", main = "Satellite Velocity offset sv_Z [E01-E36]")
for (i in index_offset_svId) {
  abline(v = i)
}

#Plot svCb
plot(data$svCb, xlab = "", xaxt='n', ylab = "Cb [s]", main = "Satellite Clock Bias offset [E01-E36]")
for (i in index_offset_svId) {
  abline(v = i)
}


