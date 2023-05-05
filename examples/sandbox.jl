#######################################################################################################
#                                           load packages
#######################################################################################################
using Pkg 
cd(@__DIR__)
Pkg.activate("..")
using Distributions
using Revise
using PTNModel
using Turing 

r = .05
n_reps = 2
θs = rand(Dirichlet(ones(4)), 10)
model = PTN(θs, .1, 20)
data = rand(model, n_reps, r)

@model function ptn(data)
    np = length(data)

    d ~ Uniform(0, .5)
    n ~ Gamma(2, 5)
    θs .~ Dirichlet(ones(4))

    data ~ PTN(θs, d, n)
end

chain = sample(ptn(data), NUTS(1000, .65), MCMCThreads(), 1000, 4)
