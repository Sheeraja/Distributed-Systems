# using Pkg
# Pkg.activate("/home/sr8685/.julia/environments/distributed_training")

# cd("/home/sr8685/distributed_systems/DistributedTraining-MNIST_Lenet")

# sinteractive --gpus=6

using CUDA, Flux, Functors, .Threads
using Metalhead, MLDatasets, MLUtils, Optimisers
using Serialization, Statistics, Random, Zygote
using ProgressMeter

_accum(x, y) = Zygote.accum(x, y)
_accum(x::T, y::T) where T <: Union{MaxPool, AdaptiveMeanPool, MeanPool, Flux.Zeros} = y
_accum(x::Function, y::Function) = x
_accum(x::Base.RefValue, y::Base.RefValue) = Ref(Zygote.accum(x.x, y.x))

include("utils.jl")
include("ddp_tasks.jl")
include("overloads.jl")
include("prepare.jl")
include("training.jl")

function getdata()
    # cifar_train = CIFAR10(:train)
    # cifar_test = CIFAR10(:test)
    mnist_train = MNIST(:train)
    mnist_test = MNIST(:test)
    # return cifar_train, cifar_test
    return mnist_train, mnist_test
end

function get_accuracy(test, m)
    # testx = test[1:100].features |> gpu
    testx = Flux.unsqueeze(test.features; dims = 3) |> gpu
    # testx = test.features |> gpu
    ypred = m(testx)

    ypred = Flux.onecold(ypred, 0:9) |> cpu;

    # acc = sum(ypred .== test[1:100].targets)
    acc = sum(ypred .== test.targets)
    return acc
end

function LeNet5(; imgsize=(28,28,1), nclasses=10) 

    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            flatten,
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end

function main()
    epochs = 5
    # cifar_train, cifar_test = getdata();
    mnist_train, mnist_test = getdata();
    model = LeNet5()
    # model = ResNet(18, nclasses=10).layers;

    nt, buf = prepare_training(model, mnist_train, devices(), Flux.Adam());
    ds_ms = train(nt, buf, epochs);

    trained_model = ds_ms[1][2] |> cpu;
    trained_model = trained_model |> gpu;
    accuracy = get_accuracy(mnist_test, trained_model);
    println("Accuracy: ", accuracy)
end







