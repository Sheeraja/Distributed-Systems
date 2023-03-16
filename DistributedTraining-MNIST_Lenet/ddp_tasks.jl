_zero(x::AbstractArray) = zero(x)
_zero(x::Base.RefValue) = Ref(_zero(x[]))
_zero(x::Function) = nothing
_zero(x::T) where T <: Union{MaxPool, AdaptiveMeanPool, Flux.Zeros} = nothing
_zero(x) = x
_zero(x::Real) = nothing

maybeT(x::NamedTuple) = x
maybeT(x) = (x,)

mywalk(f, x::T) where T = begin
  fs = fieldnames(T)
  if fs isa NTuple{N,Int} where N
    return map(f, Functors.children(x))
  end
  NamedTuple{fs}(maybeT(map(f, Functors.children(x))))
end

function destruct(o::T) where T
  Functors.fmapstructure(o, walk = mywalk) do x
    _zero(x)
  end
end

# loss(x, y) = -sum(y .* Flux.logsoftmax(x) ) ./ Float32(size(y,2))
# loss(x, y) = -sum(y .* Flux.logsoftmax(x) ) ./ Float32(size(y,2))
loss(x, y) = Flux.logitcrossentropy(x, y)

_copyto!(::Nothing, ::Nothing) = nothing
_copyto!(x::Base.RefValue, y::Base.RefValue) = Ref(_copyto!(x[], y[]))
_copyto!(x, y) = copyto!(x, y)
_copyto!(x, ::Nothing) = nothing
_copyto!(::Nothing, x) = nothing
_copyto!(x::Function, y::Function) = nothing # x
function markbuffer!(dest, src, dev)
  Functors.fmap(src, dest) do x, y
    _copyto!(y, x)
    x
  end
  synchronize()
end

function getbuffer!(dest, src, dev)
  Functors.fmap(dest, src) do x, y
    _copyto!(x, y)
    x
  end
end

function train_step(buffer, dev, m, x, y)
    local lss
    gs, _ = @device! dev begin
      y = Flux.onehotbatch(y, 0:9)
      x, y = gpu(x), gpu(y)

      gradient(m, x, y) do m, x, y
        lss = loss(m(x), y)
        lss
      end
    end
    # println("loss: $lss")
    markbuffer!(buffer[dev], gs, dev)
    gs
end

function check_nans(nt::Union{Tuple,NamedTuple})
  any(check_nans, nt)
end
check_nans(x::AbstractArray) = any(isnan, x)
check_nans(::Nothing) = false
check_nans(x) = isnan(x)

function sync_buffer(buffer)
  vals = collect(values(buffer))
  final = reduce(vals[2:end], init = vals[1]) do x,y
    Functors.fmap(x, y) do x, y
       isnothing(x) && return y
       isnothing(y) && return x
       _accum(x,y)
    end
  end

  final = Functors.fmap(final) do x
    isnothing(x) && return x
    _dodiv(x, Float32(length(vals)))
  end

  final 
end

_isapprox(x::AbstractArray, y::AbstractArray) = x ≈ y
_isapprox(::Nothing, ::Nothing) = true
# _isapprox(x::Base.RefValue, y::Base.RefValue) = _isapprox(x[], y[])
# _isapprox(x, y) = true
function ensure_synced(buffer, final)
  a = Ref{Bool}(true)
  for (dev, g) in pairs(buffer)
    Functors.fmap(g, final) do x, y
      if !_isapprox(x, y)
        a[] = false
      end
      x
    end
  end
  a[]
end



function update(opt, (dev,m), g, final, st)
  m, st = @device! dev begin
    grad = fetch(g)
    getbuffer!(grad, final, dev)
    synchronize()
    m, st = opt(m, grad, st)
    synchronize()
    m, st
  end
end
