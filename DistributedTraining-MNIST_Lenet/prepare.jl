using Optimisers

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

function getdl(i, n, data)
    i -= 1
    total = length(data)
    sub_data = 1:total
    feat = Flux.unsqueeze(data[sub_data .% n .== i].features; dims = 3)
    lbl = data[sub_data .% n .== i].targets
    
    MLUtils.DataLoader((feat, lbl); batchsize = 32, partial = false)
end

function prepare_training(model, data, devices, opt;)
    buffer = Dict()
    host = first(devices)

    zmod = destruct(model)
    # st = Optimisers.state(opt, model)
    st = Optimisers.setup(opt, model)

    for dev in devices
        Threads.@spawn begin
            buffer[dev] = @device! host begin
                gpu(zmod)
            end
        end
    end

    dls = []
    devs_ms = []
    sts = Dict()
    for (i, dev) in enumerate(devices)
        @device! dev begin
            push!(devs_ms, (dev, gpu(model)))
            sts[dev] = gpu(st)
            
            dl = getdl(i, length(devices), data)

            push!(dls, dl)
        end
    end

    (devs_ms, dls, sts), buffer
end



