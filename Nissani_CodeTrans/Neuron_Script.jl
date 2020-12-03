#=  Platform which:
a. Runs all Epochs (Train_Batch, Nr Batches each) with full neuron list;
b. Runs Validation batch; without learning, any separate nuerons grabbed. Assigns a one_vs_all list of neurons (total number being the number of classes)
c. Runs Test Batch, without learning, Grab at end;
d. Gives Error Estimate

1.
On first stage imports pre-extracted featurs from gabor filters;
Number of samples dependent on the nr C of selected classes;

2.
HUL Classifier. Uses training feature matrices S_1 to S_Bs of DIM Batch_Size*d, as imported from 1st stage
HUL Learning and HUL activity are run; Neural code vector y(:, ss_ctr)
For this Sample is calculated, and trained through time.
Any separating hyperplane subset is gathered; Error probability estimated =#

using Pkg; Pkg.activate("../Nissani_CodeTrans"); Pkg.instantiate()
include("Neuron_Learning_Cycle.jl")
using(Dates)


time_start = Dates.format(now(), "HH:MM:SS")
print(time_start)

Start_from_first_batch = 1

Display_Selected_Classes_Names = 0

Display_Dynamics = 1

Dynamics_Plot_Period = 50

if Start_from_first_batch == 1

    # classifier_features_filename = 'filename'
    # multi_epocj_classifier_results_filename = 'filename'

    epoch_nmr = 1
    trainBatchNmr = 1
    trainBatchSize = 32
    total_train_batch_nmr = trainBatchNmr*epoch_nmr

    test_batch_size = 32
    selectedClasses = "selectedClasses"

    d = "SpaceDim"

    nr_classes = length(selectedClasses)

    if Display_Selected_Classes_Names == 1
        for cc in 1:nr_classes
            class_wnid = class_wnid(selectedClasses(cc))
        end
    end

    #HUL segment parameters
    nd = 2 #number of hyperplanes per dim
    ω = 2.5 #ω < Rᵈ features space domain edge length

    θ_shift = 0
    θ_spacing = ω

    σ = 0.8 #feature analysis results
    ϵ = 0.0033
    α = 0.04 
    ϕ = 2*σ

    μᵉmode = 0
    μᵉpar = 4*ϕ

    e_start = 150 #start μ_est when tₙ > e_start

    r_start = 200
    time_var_par = 0
    ϵ_var = exp(log(0.1228)/total_train_batch_nmr)
    α_var = exp(log(0.5604)/total_train_batch_nmr)

    ϕ_var = 1

    n_N = d*nd #number of neurons in pool

    #HUL Process
    #Neuron Variables init, 1st Octant, Equally spaced, orthormal grid

    w_N = zeros(d, n_N)

    for nn = 1:d:(n_N - 1)
        w_N[:, nn[nn + d -1]] = Matrix(1.0*I, d, d)
    end

    θ_N = θ_spacing * ones(d, nd)

    for nn in 1:nd
        θ_N(:, nn) = nn*θ_N(:, nn)
    end

    θ_N = θ_N(:) + θ_shift

    c1_N = zeros(n_N, 1)
    c2_N = zers(n_N, 1)

    μ₁ᵉ = zeros(d, n_N)
    μ₂ᵉ = zeros(d, n_N)

    a_N = zeros(n_N, 1)
    y_N = zeros(n_N, 1)
    wx_N = zeros(n_N, 1)
    tₙ = zeros(n_N, 1)

    ss = 0
    b_start = 1

    for bb in b_start:total_train_batch_nmr
        if time_var_par == 1
            ϵ = ϵ*ϵ_var^(bb - 1)
            ϕ = ϕ*ϕ_var^(bb - 1)
        end
        α_sq = α^2/2
        for tt in 1:trainBatchSize
            ss = ss+1
            for nn in 1:n_N
                [w_N[:, nn], θ_N(nn), μ₁ᵉ_N[:, nn], μ₂ᵉ_N[:, nn], c1_N[nn], c2_N[nn], tₙ[nn]] =  NeuronLearningCycle!(s, w_N[:, nn], 
                                                                                                                    θ_N[nn],μ₁ᵉ_N[:, nn], μ₂ᵉ_N[:, nn], 
                                                                                                                    c1_N[nn], c2_N[nn], tₙ[nn], ϕ, ϵ, α, 
                                                                                                                    α_sq, μᵉmode, μᵉpar, r_start, E_start, d)
                wx_N[nn] = tranpose(w_N[:, nn]) * S
                y_N[nn] = ((sign(wx_N[nn] - θ_N[nn]) + 1)/2 * (wx - θ_N[nn]))
            end
        end

        last_completed = bb
        time_end = Dates.format(now(), "HH:MM:SS")
    end
end

                