"""
teInference - Inference for an edge in an elliptical coopula model.

Arguments:
              Y               (n x p) data matrix
              a, b            index of an edge for which inference is made
              methodType      number from 1-3 indicating the procedure

Method Type:
              1 (default)     uses lasso to estimate the neighborhood
                              of node a and node b
              2               uses sqrt-lasso to estimate the neighborhood
                              of node a and node b
              3               uses scalled-lasso to estimate the neighborhood
                              of node a and node b
              4               user inputs gamma_a and gamma_b
              5               OLS without model selection

Covariance Type:
              1 (default)     rank correlation
              2               pearson correlation
              3               pearson correlation after marginal
                              transformation
              4               empirical covariance X'X/n

Returns:
  (eP, eVar, ISa, ISb)

  eP   - estimated edge parameter
  eVar - estimated variance of the parameter
  ISa  - neighborhood of a
  ISb  - neighborhood of b
"""
function teInference{T<:AbstractFloat}(
  Y::Matrix{T}, a::Int, b::Int,
  methodType=1,
  covarianceType=1,
  options)

  a == b && throw(ArgumentError("a should not be equal to b"))
  n, p = size(Y)
  1 <= a <= p && throw(ArgumentError("1 <= a <= p"))
  1 <= b <= p && throw(ArgumentError("1 <= b <= p"))


  # compute covariance / correlation
  if covarianceType == 1
    covM = kendallsTau(Y)
  elseif covarianceType == 2
    covM = cor(Y)
  elseif covarianceType == 3
    covM = nonparanormalCorrelation(Y)
  elseif covarianceType == 4
    covM = cov(Y)
  else
    throw(ArgumentError("unknown covarianceType"))
  end

  I = setdiff(1:p, [a,b])

  if methodType == 1 || methodType == 2 || methodType == 3
      XX = covM(I, I);
      Xya = covM(I, a);
      Xyb = covM(I, b);

      if isscalar(lambda)
          lambda_a = lambda;
          lambda_b = lambda;
      else
          lambda_a = lambda(1);
          lambda_b = lambda(2);
      end

      if methodType == 1
          gamma_a = copulaLasso(XX, Xya, lambda_a, lambdaWeigth(:,1), 'Verbose', 0);
          gamma_b = copulaLasso(XX, Xyb, lambda_b, lambdaWeigth(:,2), 'Verbose', 0);
      else
          gamma_a = copulaDantzig(XX, Xya, lambda_a, mu);
          gamma_b = copulaDantzig(XX, Xyb, lambda_b, mu);
      end

      S_a = abs(gamma_a) > zeroThreshold ;
      S_b = abs(gamma_b) > zeroThreshold ;
      ISa = I(S_a); ISb = I(S_b);
      S = union(ISa, ISb);

      if refit
          gamma_a = covM(S, S) \ covM(S, a);
          gamma_b = covM(S, S) \ covM(S, b);
      else
          S = I;
      end

end
