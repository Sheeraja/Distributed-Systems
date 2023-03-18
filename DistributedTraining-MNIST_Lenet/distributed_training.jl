
function trainepoch(big_batches, ds_and_ms, buffer, sts)
    
    @showprogress for (_, mbs) in enumerate(big_batches)

        # mbs = first(big_batches)
        ts = []
        for ((dev,m), bs) in zip(ds_and_ms, mbs)
            gs = Threads.@spawn train_step(buffer, dev, m, bs...)
            push!(ts, Base.errormonitor(gs))
        end
        # println("loss: $lss")
        gs = ts;
        wait.(gs);

        # Step 3: Sync the buffer gradients
        final = sync_buffer(buffer)

        # move final grads to every GPU - fetch(g) has the right
        # grads for dev in it, overwrite with final
        # and optimise
        get_tasks = map(ds_and_ms, gs) do dnm, g
            t = Threads.@spawn begin
                    dev, m = dnm
                    t_opt = Threads.@spawn begin
                        @device! dev begin
                            grad = fetch(g)
                            getbuffer!(grad, final, dev)
                            synchronize()
                            st, m = Optimisers.update(sts[dev], m, grad)
                            synchronize()
                            sts[dev] = st
                        end
                        m
                    end
                (dev, fetch(Base.errormonitor(t_opt)))
            end
            Base.errormonitor(t)
        end
        ds_and_ms = fetch.(get_tasks)
    end

    ds_and_ms, sts
end

function train(nt, testdata, buffer, epochs)

    # mnist_train, mnist_test = getdata();
    ndims(testdata.features) == 3 ? (feat = Flux.unsqueeze(testdata.features, dims = 3)) : (feat = testdata.features)
    # feat = Flux.unsqueeze(mnist_test.features; dims = 3)
    
    ds_and_ms, dls, sts = nt

    big_batches = zip(dls...)

    for e in 1:epochs
        @info "epoch $e"
        
        ds_and_ms, sts = trainepoch(big_batches, ds_and_ms, buffer, sts)
        
        tempmod = ds_and_ms[1][2] |> cpu |> gpu
        ypred = tempmod(feat |> gpu) |> cpu
        lval = loss(ypred, Flux.onehotbatch(testdata.targets, 0:9))
        acc = sum(Flux.onecold(ypred, 0:9) .== testdata.targets) / length(testdata.targets)

        @show lval
        @show acc
    end

    @time trainepoch(big_batches, ds_and_ms, buffer, sts)
    ds_and_ms
end