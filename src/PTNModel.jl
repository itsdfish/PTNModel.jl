module PTNModel
    using Distributions
    
    import Distributions: ContinuousUnivariateDistribution
    import Distributions: logpdf 
    import Distributions: rand
    import Distributions: loglikelihood

    export AbstractPTN 
    export PTN 

    include("structs.jl")
    include("functions.jl")
end
