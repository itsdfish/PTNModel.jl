using LinearAlgebra

β = .7

p₁ = .10
p₂ = .05
p₃ = .85

v = [12.0,90.0,96.0]

Z = [1, 0, 0]

Q = [β * (1 - p₁)               (1 - β) * (1 - p₁)      0;
    (1 - β) * (1 - p₂) / 2      β * (1 - p₂)            (1 - β) * (1 - p₂) / 2;
    0                           (1 - β) * (1 - p₃)      β * (1 - p₃)]

R = diagm([p₁,p₂,p₃])

W = Z' * inv(I(3) - Q) * R

ev = W * v

function make_transient_matrix(β, p)
    n = length(p)
    Q = zeros(n, n)
    str = fill("",n, n)
    for c ∈ 1:n, r ∈ 1:n 
        if c == r
            Q[r,c] = β * (1 - p[r]) 
            str[r,c] = "β * (1 - p[$r])"
        elseif ((r == n) && (c == 1)) || ((r == 1) && (c == n))
            if (c == 1) || (c == n)
                Q[r,c] = 0.0
                str[r,c] = "0.0"
            else
                Q[r,c] = (1 - β) * (1 - p[r]) / 2
                str[r,c] = "(1 - β) * (1 - p[$r]) / 2"
            end
        else
            Q[r,c] = (1 - β) * (1 - p[r]) / 2 
            str[r,c] = "(1 - β) * (1 - p[$r]) / 2"
        end
    end
    return Q
end




p = [.4,.6]
Q = make_transient_matrix(β, p)
R = diagm(p)
Z = [.5,.5]

p = [p₁,p₂,p₃]
Q = make_transient_matrix(β, p)

p = [.1,.2,.3,.4]
Q = make_transient_matrix(β, p)
R = diagm(p)
sum(inv(I(4) - Q) * R, dims=2)
