

setwd('C:/Users/nsr/Documents/Hendrik/dev/nav_authenticator')


#filename <- "nav_data/10-09-2019_toe7200.csv"
filename <- "nav_data/10-09-2019_toe1800.csv"
data <- read.csv(filename, header = TRUE)


svId <- 4
ids <- data[,"svId"] == svId
X <- data[ids,]

plot(X$sp_X)
plot(X$sp_Y)
plot(X$sp_Z)
plot(X$sv_X)
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

#Plot sp_X for all satellites
plot(data$sp_X, xlab = "", ylab = "dX [m]", main = "Satellite Position offset dX")
for (i in index_offset_svId) {
  abline(v = i)
}
#Plot sv_Y for all satellites
plot(data$sv_Y, xlab = "", ylab = "delta Vy [m]", main = "Satellite Velocity delta Vy")
for (i in index_offset_svId) {
  abline(v = i)
}
#Plot svCb
plot(data$svCb, xlab = "", ylab = "delta Clock bias", main = "Satellite Clock Bias offset")
for (i in index_offset_svId) {
  abline(v = i)
}


