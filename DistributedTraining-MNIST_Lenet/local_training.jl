
function trainepoch_local(dl, model, st)
    @showprogress for (x,y) in dl
        x, y = x |> gpu, y |> gpu

        gs, _ = gradient(model, x, y) do m, x, y
            loss(m(x), y)
        end

        st, model = Optimisers.update(st, model, gs)
    end

    st, model
end


function localtrain(mod, traindata, testdata, epochs)
    ndims(traindata.features) == 3 ? (features = Flux.unsqueeze(traindata.features, dims = 3)) : (features = traindata.features)
    ndims(testdata.features) == 3 ? (feat = Flux.unsqueeze(testdata.features, dims = 3)) : (feat = testdata.features)

    targets = Flux.onehotbatch(traindata.targets, 0:9)
    dl = Flux.DataLoader((features,targets); batchsize = 32)

    mod = mod |> gpu
    st = Optimisers.setup(Flux.Adam(), mod)

    for e in 1:epochs
        @info e
        st, mod = trainepoch_local(dl, mod, st)

        tempmod = mod |> cpu
        ypred = tempmod(feat)
        lval = loss(ypred, Flux.onehotbatch(testdata.targets, 0:9))
        acc = sum(Flux.onecold(ypred, 0:9) .== testdata.targets) / length(testdata.targets)

        @show lval
        @show acc
    end
    
    @time trainepoch_local(dl, mod, st)

    mod
end