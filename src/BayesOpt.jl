module BayesOpt

const depsjl_path = joinpath(dirname(@__FILE__), "..", "deps", "deps.jl")
if !isfile(depsjl_path)
    error("BayesOpt not installed properly, run Pkg.build(\"BayesOpt\"), restart Julia and try again")
end
include(depsjl_path)

# Module initialization function
function __init__()
    # Always check your dependencies from `deps.jl`
    check_deps()
end

export KernelParameters, MeanParameters, ConfigParameters, bayes_optimization

import Base: show
struct KernelParameters
    name::Cstring
    hp_mean::NTuple{128, Cdouble}
    hp_std::NTuple{128, Cdouble}
    n_hp::Csize_t
end
function show(io::IO, mime::MIME"text/plain", o::KernelParameters)
    println(io, "$(unsafe_string(o.name)) Kernel")
    if o.n_hp > 0
        println(io, "  hyperparameters mean $(o.hp_mean[1:o.n_hp])")
        println(io, "  hyperparameters std $(o.hp_std[1:o.n_hp])")
    end
end

@enum LearningType::Cint begin
    L_FIXED
    L_EMPIRICAL
    L_DISCRETE
    L_MCMC
    L_ERROR = -1
end
export L_FIXED, L_EMPIRICAL, L_DISCRETE, L_MCMC, L_ERROR

@enum ScoreType::Cint begin
    SC_MTL
    SC_ML
    SC_MAP
    SC_LOOCV
    SC_ERROR = -1
end
export SC_MTL, SC_ML, SC_MAP, SC_LOOCV, SC_ERROR

struct MeanParameters
    name::Cstring
    coef_mean::NTuple{128, Cdouble}
    coef_std::NTuple{128, Cdouble}
    n_coef::Csize_t
end
function show(io::IO, mime::MIME"text/plain", o::MeanParameters)
    println(io, "$(unsafe_string(o.name)) Mean")
    if o.n_coef > 0
        println(io, "  coefficients mean $(o.coef_mean[1:o.n_coef])")
        println(io, "  coefficients std $(o.coef_std[1:o.n_coef])")
    end
end

mutable struct ConfigParameters
    n_iterations::Csize_t
    n_inner_iterations::Csize_t
    n_init_samples::Csize_t
    n_iter_relearn::Csize_t
    init_method::Csize_t          
    random_seed::Cint
    verbose_level::Cint
    log_filename::Cstring
    load_save_flag::Csize_t
    load_filename::Cstring
    save_filename::Cstring
    surr_name::Cstring
    sigma_s::Cdouble
    noise::Cdouble
    alpha::Cdouble
    beta::Cdouble
    sc_type::ScoreType
    l_type::LearningType
    l_all::Cint
    epsilon::Cdouble
    force_jump::Csize_t
    kernel::KernelParameters
    mean::MeanParameters
    crit_name::Cstring
    crit_params::NTuple{128, Cdouble}
    n_crit_params::Csize_t
end
ConfigParameters() = ccall((:initialize_parameters_to_default, libbayesopt), ConfigParameters, ())

function show(io::IO, mime::MIME"text/plain", o::ConfigParameters)
    println(io, "ConfigParameters")
    for field in fieldnames(ConfigParameters)
        val = getfield(o, field) 
        if field == :crit_params
            o.n_crit_params == 0 && continue
            println(io, "$field = $(val[1:o.n_crit_params])")
        end
        valtoshow = typeof(val) == Cstring ? unsafe_string(val) : val
        if typeof(val) == KernelParameters || typeof(val) == MeanParameters
            show(io, mime, val)
        else
            println(io, "$field = $valtoshow")
        end
    end
end

for func in [:set_kernel, :set_mean, :set_criteria, :set_surrogate, :set_log_file, 
             :set_load_file, :set_save_file, :set_learning, :set_score]
    @eval begin
        $(Symbol(func, "!"))(config, name) = ccall(($(string(func)), libbayesopt), Cvoid, (Ptr{ConfigParameters}, Cstring), Ref(config), name)
        export $(Symbol(func, "!"))
    end
end


function bayes_optimization(func, lb, ub, config)
    n = length(lb)
    length(ub) == n || @error("lowerbounds and upperbounds have different length.")
    optimizer = zeros(n); optimum  = Ref{Cdouble}(0)
    bofunc = (n, x, g, d) -> func(unsafe_wrap(Array, x, n))
    cfunc = @cfunction $bofunc Cdouble (Cuint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cvoid}) 
    ccall((:bayes_optimization, libbayesopt), Cint, 
          (Cint, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, 
           Ptr{Cdouble}, Ptr{Cdouble}, ConfigParameters), 
          n, cfunc, Ptr{Nothing}(0), 
          pointer(lb), pointer(ub), pointer(optimizer), optimum, config)
    (optimizer = optimizer, optimum = optimum.x)
end

end
