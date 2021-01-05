using DataFrames;
using CSV;
using HTTP;
using MLDataUtils;
using LinearAlgebra;
using JLD;

include("MultiEpochClassifier.jl")
include("RG_Unsuper\GLBNK-Unsupervised\helpers\helpers.jl")

data_link = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = DataFrame!(CSV.File(HTTP.get(data_link).body; header = false))

X, Y = MLDataUtils.load_iris()
Xs, Ys = shuffleobs((X, Y))

((cv_X, cv_Y), (test_X, test_Y)) = splitobs((Xs, Ys); at = 0.85)

@time begin
    epoch_nr = 1;
    nmr_training_batches = 10
    test_batch_size = 10
    train_batch_size = 10
    classes = ["Iris-setosa", "Iris-virginica"]
    d = 4; 
    #display_class_names = 0;
    #hyp_nmr = 2
    #Ω = 2.5
    #θshift = 0.0
    #σ = 0.8
    #μᵉmode = 0.0
    #μᵉpar = 4.0
    #E_start = 150
    #R_start = 200
    #time_var = 1
    #Φvar = 1.0
    w_N, θ, μ₁, μ₂, c1, c2, tₙ, y_N, wx_N = MultiEpochClassifier(cv_X, nmr_training_batches, 
                                                                classes, d, train_batch_size, epoch_nr);
end
@time begin

    top_n_est = 1 #Calculate performance top n
    n_Top_n = 2 #Calculate nmbr of classes to evaluate
    lambda_perr = 0.7 #Perr regularizer
    lambda_pe_top_n = 0.06 #Best regularizer factor
    nr_neurons = d*2
    nr_Classes = length(classes)

    y_N = zeros(nr_neurons, 1)
    wx_N = zeros(nr_neurons, 1)
    #μ₁, μ₂ = load("../JLD/var.jld", "miu1", "miu2") Load relevant saved values (after training) for the means
    means = hcat(μ₁, μ₂)
    separateMatrix = zeros(nr_Classes, nr_neurons)

    for nn in 1:nr_neurons
        for dd in 1:nr_Classes

            separateMatrix[dd, nn] = (sign(transpose(w_N[:, nn])*means[:, dd] - θ[nn]) + 1)/2
        end
    end

    Separate_nr = sum(separateMatrix, dims=1)
    any_separate_hp_total = count(x->(x != 0 && x != nr_Classes), Separate_nr)
    any_separate_list = findall( x -> (x!=0 && x!= nr_Classes), Separate_nr)

    one_separate_hp_total = count(x-> x==1, Separate_nr)
    one_separate_list = findall(x-> x==1, Separate_nr)

    #save("../JLD/var1.jld", "separateMatrix", separateMatrix, "separateNumber", Separate_nr, 
        #"any_separate_hp_total", any_separate_hp_total, "any_separate_list", any_separate_list,
        #"one_separate_hp_total", one_separate_hp_total, "one_separate_list", one_separate_list)

    a_N_grab_for_assign = zeros(any_separate_hp_total, train_batch_size)

    for tx in 1:train_batch_size
        for nn in 1:nr_neurons
            (y_N[nn], wx_N[nn]) = NeuronActivity(cv_X[:, nn], w_N[:, nn], θ[nn])
        end
    
        a_N_grab_for_assign[:, tx] = wx_N[any_separate_list] - θ[any_separate_list]
    end

    y = transpose(2*count(x-> x>0, a_N_grab_for_assign) - 1)

    one_vs_all_sep_array_per = zeros(nr_Classes, 2)
    one_vs_all_sep_array_pe_top_n = zeros(nr_Classes, 2)

    samples_I_assign = zeros(1, nr_Classes)
    y_columns_sum_I = zeros(nr_Classes, any_separate_hp_total)

    samples_O_assign = zeros(1, nr_Classes)
    y_column_sums_O = zeros(nr_Classes, any_separate_hp_total)

    batch_range = 1:train_batch_size-2

    for dd in 1:nr_Classes
        samples_I_assign[dd] = count(x->x==classes[dd], c1[batch_range])
        indexes_I = findall(x->x==classes[dd], c1[batch_range])
        y_column_sums_I[dd, :] = sum(y[indexes_I, :], dims=1)
        samples_O_assign[dd] = count(x->x!=classes[dd], c1[batch_range])
        indexes_O = findall(x->x!=classes[dd], c1[batch_range])
        y_column_sums_O[dd, :] = sum(y[indexes_O, :], dims=1)
    end

    for dd in 1:nr_Classes
        best_metric = -∞

        for nn in 1:any_separate_hp_total
            candidate_metric = y_column_sums_I[dd, nn] - lambda_perr*y_column_sums_O[dd, nn]

            if candidate_metric> best_metric
                best_metric = candidate_metric
                best_neuron = nn
            end
        end

        one_vs_all_sep_array_per[dd, 1] = best_neuron
        one_vs_all_sep_array_per[dd, 2] = best_metric
    end

    one_vs_all_best_neuron_perr = one_vs_all_sep_array_per[:, 1]
    one_vs_all_best_neuron_metrics_per = one_vs_all_sep_array_per[:, 2]
end

#save("../JLD/var.jld", "miu1", μ₁, "miu2", μ₂)