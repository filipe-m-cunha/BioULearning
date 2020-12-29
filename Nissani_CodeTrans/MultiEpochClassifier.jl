using MLDataUtils

include("Neuron_Learning_Cycle.jl")
include("NeuronActivity.jl")

#Will train model according to the learning rule defined in Neuron_Learning_Cycle.jl.
#Function should recieve as inputs the number of epochs, the number of training batches in the datset, the number of classes on the dataset,
#The dimension of the feature set, the size of each training batch, the training set (cv_X), the number of hyperplanes per dimension,
#And hyperparameters Ω, θshift, σ, μᵉmode, μᵉpar, E_start, R_start, time_var, Φvar

function MultiEpochClassifier(cv_X, nmr_training_batches::Int64, classes::Array{String, 1}, d::Int64, 
                            train_batch_size::Int64, epoch_nr::Int64=1, display_class_names::Int64=0, hyp_nmr::Int64=2, Ω::Float64=2.5, 
                            θshift::Float64=0.0, σ::Float64=0.8, μᵉmode::Float64=0.0, μᵉpar::Float64=4.0, 
                            E_start::Int64=150, R_start::Int64=200, time_var::Int64=1, Φvar::Float64=1.0)
    #Calculating the total number of training cycles
    total_train_batch_nmr = epoch_nr*nmr_training_batches

    nr_Classes = length(classes)

    #if display_class_names is set to one (default zero), then the class names are presented.
    if display_class_names == 1
        println(nr_Classes, "Classes, ")
        for i in 1:nr_Classes
            println(classes[i])
        end
    end

    #Initializing hyperplanes according to a grid, TODO:Change this
    θspacing = Ω/(hyp_nmr + 1)
    #Initialize ϵ according to σ
    ϵ = 0.0033*σ
    #Initialize α according to σ
    α = 0.04*σ
    #Initialize Φ according to σ
    Φ = 2*σ
    #Values of parameters ϵ, α, Φ will be changed with training, by a parameter proportional to log($parametervar)
    ϵvar = exp(log(0.1228)/total_train_batch_nmr)
    αvar = exp(log(0.5604)/total_train_batch_nmr)
    Φvar = 1
    #Calculate total number of neurons in the feature space
    nr_neurons = d*hyp_nmr
    #Initialize weigth matrix as a composition of nr_neurons/d identity matrices
    w_N = zeros(d, nr_neurons)
    for i in 1:d:nr_neurons-1
        w_N[:, i:(i+d-1)] = Diagonal(ones(d))
    end
    #Initialize the hyperplane parameter values
    θₙ = θspacing*ones(d, hyp_nmr)
    #Initialize clusters c1 and c2
    c1 = zeros(nr_neurons, 1)
    c2 = zeros(nr_neurons, 1)
    #Initialize means μ₁ and μ₂
    μ₁ = zeros(d, nr_neurons)
    μ₂ = zeros(d, nr_neurons)
    #Initialize vectors a_N, y_N, wx_N
    a_N = zeros(nr_neurons, 1)
    y_N = zeros(nr_neurons, 1)
    wx_N = zeros(nr_neurons, 1)
    #Initialize timer for each hyperplane
    tₙ = zeros(nr_neurons, 1)
    #Initialize global timer
    ss = 0
    b_start = 1
    
    for j in 1:total_train_batch_nmr
        #If specified, vary values ϵ, α, Φ proportionally to log($parametervar)
        if time_var ==1
            global ϵ *= ϵvar^(j-1)
            global α *= αvar^(j-1)
            global Φ *= Φvar^(j-1)
        end

        #Actual training section
        for n in 1: train_batch_size
            #Shuffle training set, and a batch will be extracted from it
            cv_X = shuffleobs((cv_X))
            for k in 1:nr_neurons
                #Update neuron values
                (w_N[:, k], θₙ[k], μ₁[:, k], μ₂[:, k], c1[k], c2[k], tₙ[k]) = NeuronLearningCycle(cv_X[:, s],w_N[:, s], θₙ[k], μ₁[:, k], μ₂[:, k], c1[k], c2[k], tₙ[k], Φ, ϵ, α,  μᵉmode, μᵉpar, R_start, E_start)
                (y_N[k], wx_N[k]) = NeuronActivity(cv_X[:, s], w_N[:, k], θₙ[k])
                #Update global timer value
                global ss += 1
            end
        end
    end

    return w_N, θₙ, μ₁, μ₂, c1, c2, tₙ, y_N, wx_N
end
