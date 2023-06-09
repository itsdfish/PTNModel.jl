loglikelihood(d::AbstractPTN, data::Vector{Vector{Vector{Float64}}}) = logpdf(d, data)

"""
    logpdf(dist::PTN, data, r)

# Arguments

- `dist::PTN`:
- `data`: a set of nested array in which the first level are replicates, and the second level correspond to 
    judgments a, a ∩ b, a ∪ b, a | b 
- `r=.05`: rounding factor
"""
function logpdf(dist::AbstractPTN, data::Vector{Vector{Vector{Float64}}}, r=.05)
    LL = 0.0
    for i ∈ 1:length(data)
        LL += logpdf_problem(dist, data, i, r)
    end
    return LL
end

function logpdf_problem(dist::AbstractPTN, all_data, idx, r)
    (;θs,d,n) = dist
    θ = θs[:,idx]
    data = all_data[idx]
    LL = 0.0
    for rep ∈ 1:length(data)
        # a
        θ_a = θ[1] + θ[2]
        μ_a = compute_prob(θ_a, d)
        LL += compute_log_prob(data[rep][1], θ_a, n, r)
        
        # a ∩ b
        θ_ab = θ[1]
        μ_ab = compute_prob(θ_ab, d)
        LL += compute_log_prob(data[rep][2], θ_ab, n, r)
    
        # a ∪ b
        θ_aorb = sum(θ[1:3])
        μ_aorb = compute_prob(θ_aorb, d)
        LL += compute_log_prob(data[rep][3], μ_aorb, n, r)

        # a ∣ b
        θ_b = θ[1] + θ[3]
        μ_agb =  compute_cond_prob(θ_a, θ_b, θ_ab, d)
        LL += compute_log_prob(data[rep][4], μ_agb, n, r)
    end
    return LL
end

function compute_log_prob(data, μ, n, r)
    return log(cdf(Beta(n * μ, n * (1 - μ)), data + r / 2) - 
        cdf(Beta(n * μ, n * (1 - μ)), data - r / 2))
end

function rand(dist::AbstractPTN, n_rep, r)
    return [rand_problem(dist, c, n_rep, r) for c ∈ 1:size(dist.θs,2)]
end

function rand_problem(dist::AbstractPTN, idx, n_rep, r)
    (;θs,d,n) = dist 
    θ = θs[:,idx]
    return [rand_problem(θ, d, n, r) for _ ∈ 1:n_rep]
end

function rand_problem(θs, d, n, r)
    estimates = zeros(length(θs))
    θ_a = θs[1] + θs[2]
    μ_a = compute_prob(θ_a, d)
    estimates[1] = rand(Beta(n * μ_a, n * (1 - μ_a)))

    θ_ab = θs[1]
    μ_ab = compute_prob(θ_ab, d)
    estimates[2] = rand(Beta(n * μ_ab, n * (1 - μ_ab)))

    θ_aorb = sum(θs[1:3])
    μ_aorb = compute_prob(θ_aorb, d)
    estimates[3] = rand(Beta(n * μ_aorb, n * (1 - μ_aorb)))

    θ_b = θs[1] + θs[3]
    μ_agb = compute_cond_prob(θ_a, θ_b, θ_ab, d)
    estimates[4] = rand(Beta(n * μ_agb, n * (1 - μ_agb)))

    return round_val.(estimates, r)
end

function round_val(p, r) 
    v = 1 / r 
    return round(p * v) / v
end

function compute_prob(θ, d)
    return (1 - 2 * d) * θ + d 
end

function compute_cond_prob(p_A, p_B, p_AB, d)
	t1 = (1 - 2 * d)^2 * p_AB + d * (1 - 2 * d) * (p_A + p_B) +  d^2
	t2 = (1 - 2 * d) * p_B + d
	return t1 / t2
end