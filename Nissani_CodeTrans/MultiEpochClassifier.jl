using MLDataUtils

include("Neuron_Learning_Cycle.jl")
include("NeuronActivity.jl")

function MultiEpochClassifier(epoch_nr::Int64, nmr_training_batches::Int64, test_batch_size::Int64, classes::Array{String, 1}, d::Int64, train_batch_size::Int64, display_class_names::Int64, hyp_nmr::Int64, Ω::Float64, θshift::Float64, σ::Float64, μᵉmode::Float64, μᵉpar::Float64, E_start::Int64, R_start::Int64, time_var::Int64, Φvar::Float64, cv_X)

    nr_Classes = length(classes)

    if display_class_names == 1

        println(nr_Classes, "Classes, ")

        for i in 1:nr_Classes

            println(classes[i])

        end
    end

    θspacing = Ω/(hyp_nmr + 1)

    ϵ = 0.0033*σ
    
    α = 0.04
    
    Φ = 2*σ
    
    ϵvar = exp(log(0.1228)/total_train_batch_nmr)
    
    αvar = exp(log(0.5604)/total_train_batch_nmr)
    
    Φvar = 1

    nr_neurons = d*hyp_nmr
    
    w_N = zeros(d, nr_neurons)
    
    for i in 1:d:nr_neurons-1
    
        w_N[:, i:(i+d-1)] = Diagonal(ones(d))
    
    end
    
    θₙ = θspacing*ones(d, hyp_nmr)
    
    c1 = zeros(nr_neurons, 1)
    
    c2 = zeros(nr_neurons, 1)
    
    μ₁ = zeros(d, nr_neurons)
    
    μ₂ = zeros(d, nr_neurons)
    
    a_N = zeros(nr_neurons, 1)
    
    y_N = zeros(nr_neurons, 1)
    
    wx_N = zeros(nr_neurons, 1)
    
    tₙ = zeros(nr_neurons, 1)
    
    ss = 0
    
    b_start = 1
    
    for j in 1:total_train_batch_nmr
    
        if time_var ==1
    
           global ϵ *= ϵvar^(j-1)
    
           global α *= αvar^(j-1)
    
           global Φ *= Φvar^(j-1)
    
        end


        for n in 1: train_batch_size

            cv_X = shuffleobs((cv_X))

            for k in 1:nr_neurons
        
                global ss += 1
        
                (w_N[:, k], θₙ[k], μ₁[:, k], μ₂[:, k], c1[k], c2[k], tₙ[k]) = NeuronLearningCycle(cv_X[:, s],w_N[:, s], θₙ[k], μ₁[:, k], μ₂[:, k], c1[k], c2[k], tₙ[k], Φ, ϵ, α,  μᵉmode, μᵉpar, R_start, E_start)
        
                (y_N[k], wx_N[k]) = NeuronActivity(cv_X[:, s], w_N[:, k], θₙ[k])
            end
        end
    end

    return w_N, θₙ, μ₁, μ₂, c1, c2, tₙ, y_N, wx_N
end
