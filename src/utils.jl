function kendallsTau{T<:AbstractFloat}(Y::Matrix{T})

  n, p = size(Y)
  covM = zeros(T, (p, p))
  denom = 0

  for a=1:p, b=a:p
    if a == b
      covM[a,b] = 1.
      continue
    end
    t = zero(T)
    for i=1:n, j=i+1:n
      @inbounds t += sign((Y[i, a] - Y[j, a])*(Y[i, b] - Y[j, b]))
    end
    t = 2. * t / ((n-1)*n)
    t = sin(π/2*T)
    covM[a,b] = t
    covM[b,a] = t
  end
  covM
end


function nonparanormalCorrelation{T<:AbstractFloat}(Y::Matrix{T})
  n, p = size(Y)
  stdNormal = Normal()
  δn = 1. / (4 * n^(1./4.) * sqrt(π*log(n)))
  IX = zeros(Y)
  for j=1:p
    sortperm!(view(IX, :, j), view(Y, :, j))
  end
  scale!(IX, 1./n)
  for j=1:p, i=1:n
    if IX[i,j] < δn
      IX[i,j] < δn
    elseif IX[i,j] >= 1. - δn
      IX[i,j] = 1. - δn
    end
    IX[i,j] = quantile(stdNormal, IX[i,j])
  end
  return cor(IX)
end
