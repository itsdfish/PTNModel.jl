abstract type AbstractPTN <: ContinuousUnivariateDistribution end

"""
    PTN{T,T1,T2}

Probability theory + noise model for events 
- a
- a ∩ b
- a ∪ b
- a | b 

# Fields 
- `θs::Array{T,2}`: true subjective probability in which each column corresponds to events a ∩ b, a ∩ ¬b, ¬a ∩ b, ¬a ∩ ¬b 
- `d::T1`: miscount probability 
- `n::T2`: the concentration of a beta distribution
"""
struct PTN{T,T1,T2} <: AbstractPTN
    θs::Array{T,2}
    d::T1
    n::T2
end
