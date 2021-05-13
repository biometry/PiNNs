#'---
#'title: "Run PRELES model for some Boreal sites"
#'author: Francesco Minunno
#'email: francesco.minunno@helsinki.fi
#'date: 29 April 2015
#'---
#' 

#'Synopsis: This short tutorial shows how to run PRELES for four Scots Pine forests in Finland.
#' 
#' Loading data
#' ===============================

#' Load the datasets for the four sites: 'Hyytiälä', 'Sodankyl??', 'Alkkia' and 'Kalevansuo'.
#' Data cover two years and each dataframe contains vectors of the climatic variables (PAR, TAir, VPD, Precip, CO2, fAPAR) and the observed data (GPP and ET).

load('Boreal_sites.rdata')
ls()
head(s1)

#' Run PRELES for the 4 Boreal sites
#' ===============================

#' Run PRELES through the R function. If the parameter vector is not specified the default parameter values will be used.

library(Rpreles)
preles_s1 <- PRELES(PAR=s1$PAR, TAir=s1$TAir, VPD=s1$VPD, 
	Precip=s1$Precip, CO2=s1$CO2, fAPAR=s1$fAPAR)
preles_s2 <- PRELES(PAR=s2$PAR, TAir=s2$TAir, VPD=s2$VPD, 
	Precip=s2$Precip, CO2=s2$CO2, fAPAR=s2$fAPAR)
preles_s3 <- PRELES(PAR=s3$PAR, TAir=s3$TAir, VPD=s3$VPD, 
	Precip=s3$Precip, CO2=s3$CO2, fAPAR=s3$fAPAR)
preles_s4 <- PRELES(PAR=s4$PAR, TAir=s4$TAir, VPD=s4$VPD, 
	Precip=s4$Precip, CO2=s4$CO2, fAPAR=s4$fAPAR)

#' Plotting results
#' ===============================

par(mfrow=c(2,1),oma=c(0,0,2,0))
plot(s1$GPPobs,col=2,xlab='',ylab='tC ha?????', main='GPP')
lines(preles_s1$GPP,col=3)
plot(s1$ETobs,xlab='',ylab='mm ha?????', main='ET')
lines(preles_s1$ET,col=4)
title(main="Site 1",outer=T)

par(mfrow=c(2,1),oma=c(0,0,2,0))
plot(s2$GPPobs,col=2,xlab='',ylab='tC ha?????', main='GPP')
lines(preles_s2$GPP,col=3)
plot(s2$ETobs,xlab='',ylab='mm ha?????', main='ET')
lines(preles_s2$ET,col=4)
title(main="Site 2",outer=T)

par(mfrow=c(2,1),oma=c(0,0,2,0))
plot(s3$GPPobs,col=2,xlab='',ylab='tC ha?????', main='GPP')
lines(preles_s3$GPP,col=3)
plot(s3$ETobs,xlab='',ylab='mm ha?????', main='ET')
lines(preles_s3$ET,col=4)
title(main="Site 3",outer=T)

par(mfrow=c(2,1),oma=c(0,0,2,0))
plot(s4$GPPobs,col=2,xlab='',ylab='tC ha?????', main='GPP')
lines(preles_s4$GPP,col=3)
plot(s4$ETobs,xlab='',ylab='mm ha?????', main='ET')
lines(preles_s4$ET,col=4)
title(main="Site 4",outer=T)


