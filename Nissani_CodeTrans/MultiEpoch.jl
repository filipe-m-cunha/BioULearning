include("Neuron_Learning_Cycle.jl")
include("NeuronActivity.jl")


function normalize(w::Array{Float64,2})
    wnew = zeros(size(w)[1], size(w)[2])
    for i in 1:size(w)[2]
        wnew[:, i] = w[:, i]./sum(w[:, i])
    end
    return wnew
end

function hyperplane_inicialization(d::Int64, nmr_hyp::Int64, orientation = "random", Ω::Float64 = 2.0, θshift=0.1)

    nr_neurons = d*nmr_hyp

    if (orientation == "random")
        w = normalize(rand(d, nr_neurons))
        θ = Ω.*rand(nr_neurons)

    elseif (orientation == "grid")
        w = zeros(d, nr_neurons)
        for i in 1:d:nr_neurons-1
            w[:, i:(i+d-1)] = Diagonal(ones(d))
        end
        θspacing = Ω/(nmr_hyp + 1)
        θ = θspacing*ones(d, nmr_hyp)
        for i in 1:nmr_hyp
            θ[:, i] = i*θ[:, i]
        end
        θ = reshape(θ, size(θ)[1]*size(θ)[2], 1)
        θ = θ .+ θshift
    else
        @assert false "orientation has to be either random or grid!"
    end
    return w, θ
end

function hyperplane_additional_shift(training_set, w, θ, ϕ, Ω)

    indices = rand(1:size(training_set)[1])
    close = zeros(size(w)[2])
    for i in 1:size(w)[2]
        for j in indices
            if (transpose(w[:, i])*training_set[j] ≥ (θ[i] - ϕ)) & (transpose(w[:, i])*training_set[j]  ≤ (θ[i] + ϕ))
                close[i] = 1
                break
            end
        end
        if close[i] == 0
            w[:, i] = rand(size(w)[2])
            w[:, i] = w[:, i]./sum(w[:, i])
            θ[i] = Ω*rand()
        end
    end
    return w, θ
end

function MultiEpoch(training_set, nmr_training_batches::Int64, d::Int64,
                    size_training_batch::Int64, nmr_epochs::Int64, nmr_hyp::Int64=8,
                    Ω::Float64=2,  σ::Float64=0.8, μᵉmode::Float64=0.0, μᵉpar::Float64=4.0, 
                    E_start::Int64=10, R_start::Int64=20, initial_orientation="random")

    nr_neurons = d*nmr_hyp
    θshift = 1
    θspacing = Ω/(nmr_hyp + 1)
    ϵ = 2*σ
    α = 0.5
    ϕ = 2.0

    w_N, θ = hyperplane_inicialization(d, nmr_hyp, initial_orientation, Ω, θshift)

    #Initialize cumulative weights c1 and c2 (for weighted average to define the mean)
    c1 = zeros(nr_neurons, 1)
    c2 = zeros(nr_neurons, 1)
    #Initialize means μ₁ and μ₂ at zero
    μ₁ = zeros(d, nr_neurons)
    μ₂ = zeros(d, nr_neurons)
    #Initialize vectors y_N, wx_N
    y_N = zeros(nr_neurons, 1)
    wx_N = zeros(nr_neurons, 1)
    #Initialize timer for each hyperplane
    tₙ = zeros(nr_neurons, 1)
    #Initialize global timer
    ss = 0
    b_start = 1

    #TODO: Vary alpha and epsilon with timer

    for i in 1:nmr_training_batches*nmr_epochs
        for j in 1:size_training_batch
            ss = ss + 1
            for k in 1:nr_neurons
                #println("w_N: ", w_N[:, k])
                #println("theta: ", θ[k])
                (w_N[:, k], θ[k], μ₁[:, k], μ₂[:, k], c1[k], c2[k], tₙ[k]) = NeuronLearningCycle(training_set[j, :],w_N[:, k], θ[k], μ₁[:, k], μ₂[:, k], c1[k], c2[k], tₙ[k], ϕ, ϵ, α,  μᵉmode, μᵉpar, R_start, E_start)
                (wx_N[k], y_N[k]) = NeuronActivity(training_set[j, :], w_N[:, k], θ[k])
                if count(x -> (isnan(x)), w_N) > 0
                    println("Failed at neuron:")
                    println(k)
                    println("x: ", training_set[:, j])
                    #println("With values:")
                    println("w_N: ", w_N[:, k])
                    #println("theta: ", θ[k])
                    break
                end
            end
            if count(x -> (isnan(x)), w_N) > 0
                println("Failed at ss: ")
                println(ss)
                break
            end
        end
    end

    return w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N
end
