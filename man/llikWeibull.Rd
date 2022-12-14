% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/llik.R
\name{llikWeibull}
\alias{llikWeibull}
\title{log likelihood and derivatives for Weibull distribution}
\usage{
llikWeibull(x, shape, scale, full = FALSE)
}
\arguments{
\item{x}{variable distributed by a Weibull distribution}

\item{shape, scale}{shape and scale parameters, the latter defaulting to 1.}

\item{full}{Add the data frame showing x, mean, sd as well as the
fx and derivatives}
}
\value{
data frame with \code{fx} for the log pdf value of with \code{dProb}
that has the derivatives with respect to the \code{prob} parameters at
the observation time-point
}
\description{
log likelihood and derivatives for Weibull distribution
}
\examples{

llikWeibull(1, 1, 10)

}
\author{
Matthew L. Fidler
}
