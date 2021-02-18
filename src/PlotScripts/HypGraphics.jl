using MLDataUtils;
using Printf;
using DataFrames;
using StatsPlots;

include("../Model/MultiEpoch.jl")
include("../ModelError/ErrFunction.jl")

function plot_hyperparameter_variation(parameter, min::Float64, max::Float64, niter::Int64=100, local_save::Bool=true)

    X, Y = MLDataUtils.load_iris()
    Xs, Ys = shuffleobs((X, Y))

    epoch_nr = 100;
    nmr_training_batches = 1
    train_batch_size = 150
    d = 4
    hyp_nmr = 2
    μᵉmode = 0.0
    μᵉpar = 4.0

    #Parameters to evaluate
    α = 0.012
    ϵ = 0.00005
    ϕ = 2.0
    Ω = 5.0
    σ = 0.8
    E_start = 200
    R_start = 300
    initial_orientation = "random"
    vary = 0
    val = range(min, stop=max, length=niter)
    acc_val = zeros(niter)
    if (parameter == "α") | (parameter == "alpha") | (parameter == "Alpha")
        for j in 1:niter
            acc = zeros(5)
            for i in 1:5
                (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(Xs, nmr_training_batches, d, 
                                                                train_batch_size, epoch_nr,
                                                                hyp_nmr, Ω, ϵ, val[j], ϕ, σ, μᵉmode,
                                                                μᵉpar, E_start, R_start,
                                                                initial_orientation, vary);
                acc_i = get_model_acc(Xs, Ys, w_N, θ)
                acc[i] = acc_i
            end
            global acc_val[j] = maximum(acc)
        end
    elseif (parameter == "ϵ") | (parameter == "epsilon") | (parameter == "Epsilon")
        for j in 1:niter
            acc = zeros(5)
            for i in 1:5
                (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(Xs, nmr_training_batches, d, 
                                                                train_batch_size, epoch_nr,
                                                                hyp_nmr, Ω, val[j], α, ϕ, σ, μᵉmode,
                                                                μᵉpar, E_start, R_start,
                                                                initial_orientation, vary);
                acc_i = get_model_acc(Xs, Ys, w_N, θ)
                acc[i] = acc_i
            end
            global acc_val[j] = maximum(acc)
        end
    elseif (parameter == "ϕ") | (parameter == "phi") | (parameter == "Phi")
        for j in 1:niter
            acc = zeros(5)
            for i in 1:5
                (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(Xs, nmr_training_batches, d, 
                                                                train_batch_size, epoch_nr,
                                                                hyp_nmr, Ω, ϵ, α, val[j], σ, μᵉmode,
                                                                μᵉpar, E_start, R_start,
                                                                initial_orientation, vary);
                acc_i = get_model_acc(Xs, Ys, w_N, θ)
                acc[i] = acc_i
            end
            global acc_val[j] = maximum(acc)
        end
    elseif (parameter == "Ω") | (parameter == "omega") | (parameter == "Omega")
        for j in 1:niter
            acc = zeros(5)
            for i in 1:5
                (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(Xs, nmr_training_batches, d, 
                                                                train_batch_size, epoch_nr,
                                                                hyp_nmr, val[j], ϵ, α, ϕ, σ, μᵉmode,
                                                                μᵉpar, E_start, R_start,
                                                                initial_orientation, vary);
                acc_i = get_model_acc(Xs, Ys, w_N, θ)
                acc[i] = acc_i
            end
            global acc_val[j] = maximum(acc)
        end
    elseif (parameter == "σ") | (parameter == "sigma") | (parameter == "Sigma")
        for j in 1:niter
            acc = zeros(5)
            for i in 1:5
                (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(Xs, nmr_training_batches, d, 
                                                                train_batch_size, epoch_nr,
                                                                hyp_nmr, Ω, ϵ, α, ϕ, val[j], μᵉmode,
                                                                μᵉpar, E_start, R_start,
                                                                initial_orientation, vary);
                acc_i = get_model_acc(Xs, Ys, w_N, θ)
                acc[i] = acc_i
            end
            global acc_val[j] = maximum(acc)
        end
    else
        @assert false "parameter has to be either alpha, epsilon, phi, omega or sigma!"
    end

    df = DataFrame(A = val, B = acc_val)
    plotd = @df df StatsPlots.plot(:A, :B);
    
    title!("Variation of the accuracy with changes for the $(parameter) parameter")
    xlabel!(parameter)
    ylabel!("Accuracy")

    if local_save
        savefig(plotd,"$(parameter)_plot.png")
    end
end

plot_hyperparameter_variation("α", 0.0001, 0.02, 1000)

