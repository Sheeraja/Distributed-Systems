

function train(nt, buffer, epochs)

    mnist_train, mnist_test = getdata();
    feat = Flux.unsqueeze(mnist_test.features; dims = 3)
    
    ds_and_ms, dls, sts = nt

    big_batches = zip(dls...)

    for e in 1:epochs
        @info "epoch $e"
        @showprogress for (j, mbs) in enumerate(big_batches)

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
            # @sync begin
            #     for (i, (dnm, g)) in enumerate(zip(ds_and_ms, gs))
            #         dev, m = dnm
            #         # println("Before_$dev:")
            #         # println(Flux.params(m[1][1])[1][:,1,1,1])
            #         @async begin
            #             device!(i-1)
            #             grad = fetch(g)     
            #             getbuffer!(grad, final, dev)
            #             synchronize()
            #             st, m = Optimisers.update(sts[dev], m, grad)
            #             synchronize()
            #             sts[dev] = st
            #         end
            #     end
            # end
            # for (dev, m) in ds_and_ms
            #     println("After_$dev:")
            #     println(Flux.params(m[1][1])[1][:,1,1,1])
            # end
        end
        
        tempmod = ds_and_ms[1][2] |> cpu
        ypred = tempmod(feat)
        lval = loss(ypred, Flux.onehotbatch(mnist_test.targets, 0:9))
        acc = sum(Flux.onecold(ypred, 0:9) .== mnist_test.targets) / length(mnist_test.targets)

        @show lval
        @show acc
    end
    ds_and_ms
end

# x = rand(Float32, 3, 3)
# y = rand(Float32, 3, 3)

# res = []

# @sync begin
#     @async begin
#         device!(0)
#         push!(res, gpu(x))
#     end
#     @async begin
#         device!(1)
#         push!(res, gpu(y))
#     end
# end

# @async begin
#     device!(0)
#     println(res[2])
# end