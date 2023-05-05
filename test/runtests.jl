using SafeTestsets

# @safetestset "recovery" begin
#     using PTNModel
#     using Test
#     using Random 
#     using Distributions
#     Random.seed!(684)

#     r = .05
#     d = .1
#     n = 10.0
#     n_reps = 10_000
#     θs = rand(Dirichlet(ones(4)),1)

#     ds = range(d * .8, d * 1.2, length = 100)
#     ns = range(n * .8, n * 1.2, length = 100)

#     model = PTN(θs, d, n)
#     data = rand(model, n_reps, r)

#     LLs = map(d -> logpdf(PTN(θs, d, n), data, r), ds)
#     _,idx = findmax(LLs)
#     @test ds[idx] ≈ d atol = 1e-2

#     LLs = map(n -> logpdf(PTN(θs, d, n), data, r), ns)
#     _,idx = findmax(LLs)
#     @test ns[idx] ≈ n atol = 1
# end

@safetestset "compute_log_prob" begin 
    using PTNModel
    using Test
    using Distributions
    using PTNModel: compute_log_prob

    r = .05
    μ = .5
    n = 2.0
    n_reps = 10_000

    log_prob = compute_log_prob(.5, μ, n, r)
    @test log_prob ≈ log(r)

    r = .05
    μ = .5
    n = 10.0
    n_reps = 10_000

    true_log_prob = cdf(Beta(5,5), .525) - cdf(Beta(5,5), .475) |> log
    log_prob = compute_log_prob(.5, μ, n, r)
    @test log_prob ≈ true_log_prob
end