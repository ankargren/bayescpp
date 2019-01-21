#include "bayescpp.h"

#ifndef BAYESCPP_HS_H
#define BAYESCPP_HS_H
arma::mat sample_hs_local(const arma::mat& local, const arma::mat& pi, double global) {
  arma::uword n_param = local.n_elem;

  // Generate the f(u|x) part (Damien et al., sec 3.2)
  arma::mat eta = arma::pow(local, -2.0);
  arma::mat u1 = arma::mat(n_param, 1, arma::fill::zeros);
  std::generate(u1.begin(), u1.end(), ::unif_rand);
  u1 %= 1/(1+eta);
  arma::mat temp_upper = (1-u1)/u1;

  // Generate from f(x|u) (truncated exponential)
  arma::mat temp_exp = arma::pow(pi, 2.0) / (2.0 * std::pow(global, 2.0));
  arma::mat F_upper = 1.0 - arma::exp(-temp_exp % temp_upper);
  arma::uvec temp_small = arma::find(F_upper < 0.0001); // Numerical stability
  F_upper.elem(temp_small).fill(0.0001);

  // Inverse transform sampling
  arma::mat u2 = arma::mat(n_param, 1, arma::fill::zeros);
  std::generate(u2.begin(), u2.end(), ::unif_rand);
  u2 %= F_upper;
  eta = -arma::log(1-u2)/temp_exp;
  arma::mat local_out = arma::pow(eta, -0.5);

  return local_out;
}

double sample_hs_global(const arma::mat& local, const arma::mat& pi, double global, double global_scale2 = 1.0) {
  arma::uword n_param = local.n_elem;

  double temp_b = arma::accu(arma::pow(pi/local, 2.0) / 2.0);
  double eta = std::pow(global, -2.0);
  double u1 = R::runif(0, 1.0/(1.0+eta*global_scale2));
  double temp_upper = (1.0-u1)/u1;

  double F_upper = R::pgamma(temp_upper, (n_param+1)/2.0, 1.0/temp_b, 1, 0);
  F_upper = std::max(F_upper, 0.00000001);
  double u2 = R::runif(0, F_upper);
  eta = R::qgamma(u2, (n_param+1)/2.0, 1.0/temp_b, 1, 0);
  double global_out = std::pow(eta, -0.5);
  return global_out;
}

#endif

