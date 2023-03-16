module FluxDistributed

using Flux, CUDA
using Metalhead
using BSON, Zygote
using Distributed
using Dates, DataSets
using Functors, Optimisers
using Requires, Logging

export minibatch, train_solutions, syncgrads
export prepare_training

include("preprocess.jl")
include("utils.jl")
include("overloads.jl")
# include("ddp_tasks.jl")



end # module
