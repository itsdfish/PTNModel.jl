module PTNModel
    using Distributions
    
    import Distributions: ContinuousUnivariateDistribution
    import Distributions: logpdf 
    import Distributions: rand

    export AbstractPTN 
    export PTN 


    include("structs.jl")
    include("functions.jl")
end
