

#setwd('D:/Documents/SCHOOL/Ingenieurswetenschappen - Computerwetenschappen/KUL/Jaar II/Industriele Stage - ESA/dev/nav_authenticator')
setwd('C:/Users/nsr/Documents/Hendrik/dev/nav_authenticator')

filename6 <- "nav_data/week2_v4_accCheck_toe600.csv"
filename12 <- "nav_data/week2_v4_accCheck_toe1200.csv"
filename18 <- "nav_data/week2_v4_accCheck_toe1800.csv"
filename24 <- "nav_data/week2_v4_accCheck_toe2400.csv"
filename30 <- "nav_data/week2_v4_accCheck_toe3000.csv"
filename36 <- "nav_data/week2_v4_accCheck_toe3600.csv"
filename42 <- "nav_data/week2_v4_accCheck_toe4200.csv"
filename48 <- "nav_data/week2_v4_accCheck_toe4800.csv"
filename54 <- "nav_data/week2_v4_accCheck_toe5400.csv"
filename60 <- "nav_data/week2_v4_accCheck_toe6000.csv"
filename66 <- "nav_data/week2_v4_accCheck_toe6600.csv"
filename72 <- "nav_data/week2_v4_accCheck_toe7200.csv"

data6 <- read.csv(filename6, header = TRUE)
data12 <- read.csv(filename12, header = TRUE)
data18 <- read.csv(filename18, header = TRUE)
data24 <- read.csv(filename24, header = TRUE)
data30 <- read.csv(filename30, header = TRUE)
data36 <- read.csv(filename36, header = TRUE)
data42 <- read.csv(filename42, header = TRUE)
data48 <- read.csv(filename48, header = TRUE)
data54 <- read.csv(filename54, header = TRUE)
data60 <- read.csv(filename60, header = TRUE)
data66 <- read.csv(filename66, header = TRUE)
data72 <- read.csv(filename72, header = TRUE)

get_std_vector <- function(data1) {
  
  subInd1 <- seq(0,length(data1$svId),30)
  dataSub1 <- data1[subInd1,]
  wrong_ind1 <- abs(dataSub1$sp_X) > 1000
  dataSub1_ <- dataSub1[!wrong_ind1,]
  std_vec <- apply(dataSub1_,2,sd)
  
}

std1 <- get_std_vector(data6)
std2 <- get_std_vector(data12)
std3 <- get_std_vector(data18)
std4 <- get_std_vector(data24)
std5 <- get_std_vector(data30)
std6 <- get_std_vector(data36)
std7 <- get_std_vector(data42)
std8 <- get_std_vector(data48)
std9 <- get_std_vector(data54)
std10 <- get_std_vector(data60)
std11 <- get_std_vector(data66)
std12 <- get_std_vector(data72)


stds <- data.frame(rbind(std1, std2, std3, std4, std5, std6, std7, std8, std9, std10, std11, std12))
toe <- c(600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600, 7200)
stds$toe <- toe

plot(toe, stds$sp_X, type='o', xlim = c(0,7800), ylim = c(0, 400), xaxp = c(0,7800,13), yaxp = c(0,400,8),
     xlab = "ToE [s]", ylab = "STD(sp) [m]", main = "Progression of STD of error in position with increasing ToE", col = "red")
lines(toe, stds$sp_Y, type='o', col = "blue")
lines(toe, stds$sp_Z, type='o', col = "green")
grid()
legend("topleft", legend = c("STD(sp_X)", "STD(sp_Y)", "STD(sp_Z)"), col = c("red", "blue", "green"), lty = c(1,1,1), pch = c(1,1,1))

plot(toe, stds$sv_X, type='o', xlim = c(0,7800), ylim = c(0, 0.05), xaxp = c(0,7800,13), yaxp = c(0,0.05,5),
     xlab = "ToE [s]", ylab = "STD(sv) [m/s]", main = "Progression of STD of error in velocity with increasing ToE", col = "red")
lines(toe, stds$sv_Y, type='o', col = "blue")
lines(toe, stds$sv_Z, type='o', col = "green")
grid()
legend("topleft", legend = c("STD(sv_X)", "STD(sv_Y)", "STD(sv_Z)"), col = c("red", "blue", "green"), lty = c(1,1,1), pch = c(1,1,1))

plot(toe, stds$svCb, type='o', xlim = c(0,7800), ylim = c(0, 6e-10), xaxp = c(0,7800,13), yaxp = c(0,6e-10,6),
     xlab = "ToE [s]", ylab = "STD(svCb) [s]", main = "Progression of STD of error in clock bias with increasing ToE")
grid()




#Fit models
lm_spX <- lm(sp_X ~ toe + I(toe^2), data = stds)
summary(lm_spX)
lm_spX$coefficients
spX_pred <- predict(lm_spX, data.frame(toe))
plot(toe, stds$sp_X)
lines(toe, spX_pred, col = "red")

lm_spY <- lm(sp_Y ~ toe + I(toe^2), data = stds)
lm_spY$coefficients
lm_spZ <- lm(sp_Z ~ toe + I(toe^2), data = stds)
lm_spZ$coefficients

lm_svX <- lm(sv_X ~ toe + I(toe^2), data = stds)
lm_svX$coefficients
lm_svY <- lm(sv_Y ~ toe + I(toe^2), data = stds)
lm_svY$coefficients
lm_svZ <- lm(sv_Z ~ toe + I(toe^2), data = stds)
lm_svZ$coefficients

lm_svCb <- lm(svCb ~ toe + I(toe^2), data = stds)
lm_svCb$coefficients
svCb_pred <- predict(lm_svCb, data.frame(toe))
plot(toe, stds$svCb)
lines(toe, svCb_pred)
