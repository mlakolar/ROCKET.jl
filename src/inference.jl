struct ROCKETOptions
  λ::Float64
  zeroThreshold::Float64
  refit::Bool
  optionsCD::CDOptions
  optionsScaledLasso::ScaledLassoOptions
end

ROCKETOptions(;
  λ::Float64=0.3,
  zeroThreshold::Float64=1e-4,
  refit::Bool=true,
  optionsCD::CDOptions=CDOptions(),
  optionsScaledLasso::ScaledLassoOptions=ScaledLassoOptions()) =
     ROCKETOptions(λ, zeroThreshold, refit, optionsCD, optionsScaledLasso)


function _stdColumn!{T<:AbstractFloat}(out::Vector{T}, X::AbstractMatrix{T})

  n, p = size(X)
  p == length(out) || throw(DimensionMismatch())

  for j=1:p
    t = zero(T)
    for i=1:n
      @inbounds t += X[i, j]^2.
    end
    @inbounds out[j] = sqrt(t / n)
  end

  out
end

# hOmega = inv(ɛ'ɛ/n)
# eP = hOmega[1,2]
# eVar = (hOmega[1,1]*hOmega[2,2] + hOmega[1,2]*hOmega[1,2]) / n
function _TEinv22{T<:AbstractFloat}(ɛ::Matrix{T})
  n, p = size(ɛ)
  p == 2 || throw(ArgumentError())

  a = zero(T)
  b = zero(T)
  c = zero(T)
  for i=1:n
    a += ɛ[i, 1] * ɛ[i, 1]
    c += ɛ[i, 2] * ɛ[i, 2]
    b += ɛ[i, 1] * ɛ[i, 2]
  end

  dt = a*c - b*b
  eP = -b / dt * n
  eVar = (a * c + b * b) / dt / dt * n
  eP, eVar
end

function _teInferenceGaussian{T<:AbstractFloat}(
  Y::Matrix{T}, a::Int, b::Int, S::Vector{Int})

  n, p = size(Y)
  x = view(Y, :, S)
  y = view(Y, :, [a,b])
  β = x \ y
  ɛ = x * β
  @. ɛ = y - ɛ
  _TEinv22(ɛ)
end

function _teInferenceGaussian{T<:AbstractFloat}(
  Y::Matrix{T}, a::Int, b::Int,
  γa::Union{SparseIterate{T}, Vector{T}},
  γb::Union{SparseIterate{T}, Vector{T}})

  n, p = size(Y)
  I = setdiff(1:p, [a,b])
  x = view(Y, :, I)
  y = view(Y, :, [a,b])

  ɛ = zeros(T, (n, 2))
  @inbounds for i=1:n
    ɛ[i, 1] = HD._row_A_mul_b(x, γa, i)
    ɛ[i, 2] = HD._row_A_mul_b(x, γb, i)
  end
  @. ɛ = y - ɛ
  _TEinv22(ɛ)
end



# Method Type:
#               1 (default)     uses lasso to estimate the neighborhood
#                               of node a and node b
#               2               uses sqrt-lasso to estimate the neighborhood
#                               of node a and node b
#               3               uses scalled-lasso to estimate the neighborhood
#                               of node a and node b
#               4               OLS without model selection
function _teInferenceGaussian{T<:AbstractFloat}(
  Y::Matrix{T}, a::Int, b::Int,
  methodType::Int,
  options::ROCKETOptions=ROCKETOptions())

  n, p = size(Y)
  I = setdiff(1:p, [a,b])

  λ = options.λ
  τ0 = options.zeroThreshold

  if methodType in [1, 2, 3]
    # lasso
    x = view(Y, :, I)
    ya = view(Y, :, a)
    yb = view(Y, :, b)

    scaleX = zeros(p-2)
    _stdColumn!(scaleX, x)
    scale!(scaleX, λ)

    g = AProxL1(scaleX)
    γa = SparseIterate(p-2)
    γb = SparseIterate(p-2)
    if methodType == 1
      fa = CDLeastSquaresLoss(ya, x)
      fb = CDLeastSquaresLoss(yb, x)
      coordinateDescent!(γa, fa, g, options.optionsCD)
      coordinateDescent!(γb, fb, g, options.optionsCD)
    elseif methodType == 2
      fa = CDSqrtLassoLoss(ya, x)
      fb = CDSqrtLassoLoss(yb, x)
      coordinateDescent!(γa, fa, g, options.optionsCD)
      coordinateDescent!(γb, fb, g, options.optionsCD)
    else # if methodType == 3
      scaledLasso!(γa, x, ya, scaleX, options.optionsScaledLasso)
      scaledLasso!(γb, x, yb, scaleX, options.optionsScaledLasso)
    end

    if options.refit
      Sa = find(x -> abs(x) > τ0, γa)
      Sb = find(x -> abs(x) > τ0, γb)
      S = union(I[Sa], I[Sb])

      return _teInferenceGaussian(Y, a, b, S)
    else
      return _teInferenceGaussian(Y, a, b, γa, γb)
    end
  elseif methodType == 4
    return _teInferenceGaussian(Y, a, b, I)
  else
    throw(ArgumentError("Unknown methodType"))
  end
end


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
              4               OLS without model selection

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
  methodType::Int=1,
  covarianceType::Int=1,
  options::ROCKETOptions=ROCKETOptions())

  a == b && throw(ArgumentError("a should not be equal to b"))
  n, p = size(Y)
  1 <= a <= p || throw(ArgumentError("1 <= a <= p"))
  1 <= b <= p || throw(ArgumentError("1 <= b <= p"))

  if covarianceType != 4
    throw(ArgumentError("unknown covarianceType"))
  end
  return _teInferenceGaussian(Y, a, b, methodType, options)

  # # compute covariance / correlation
  # if covarianceType == 1
  #   covM = kendallsTau(Y)
  #   throw(ArgumentError("unknown covarianceType"))
  # elseif covarianceType == 2
  #   covM = cor(Y)
  #   throw(ArgumentError("unknown covarianceType"))
  # elseif covarianceType == 3
  #   covM = nonparanormalCorrelation(Y)
  #   throw(ArgumentError("unknown covarianceType"))
  # elseif covarianceType == 4
  #   return _teInferenceGaussian(Y, a, b, methodType, options)
  # else
  #   throw(ArgumentError("unknown covarianceType"))
  # end
end
