using JLD
using LinearAlgebra
using Infinity

include("NeuronActivity.jl")

@time begin

    top_n_est = 1 #Calculate performance top n
    n_Top_n = 2 #Calculate nmbr of classes to evaluate
    lambda_perr = 0.7 #Perr regularizer
    lambda_pe_top_n = 0.06 #Best regularizer factor

    μ₁, μ₂ = nothing #Begin values at Null

    y_N = zeros(nr_neurons, 1)
    wx_N = zeros(nr_neurons, 1)
    μ₁, μ₂ = load("../JLD/var.jld", "miu1", "miu2") #Load relevant saved values (after training) for the means

    separateMatrix = zeros(nr_Classes, nr_neurons)

    for nn in 1:nr_neurons
        for dd in 1:nr_Classes

            separateMatrix[dd, nn] = (sign(transpose(w_N[:, nn])*means[:, dd] - θₙ[nn]) + 1)/2
        end
    end

    Separate_nr = sum(separateMatrix, dims=1)
    any_separate_hp_total = count(x->(x != 0 & x != nr_Classes), Separate_nr)
    any_separate_list = findall( x -> (x!=0 & x!= nr_Classes), Separate_nr)

    one_separate_hp_total = count(x-> x==1, Separate_nr)
    one_separate_list = findall(x-> x==1, Separate_nr)

    save("../JLD/var1.jld", "separateMatrix", separateMatrix, "separateNumber", Separate_nr, 
        "any_separate_hp_total", any_separate_hp_total, "any_separate_list", any_separate_list,
        "one_separate_hp_total", one_separate_hp_total, "one_separate_list", one_separate_list)

    a_N_grab_for_assign = zeros(any_separate_hp_total, train_batch_size)

    for tx in 1:train_batch_size
        for nn in 1:nr_neurons
            (y_N[nn], wx_N[nn]) = NeuronActivity(cv_X[nn], w_N[:, nn], θₙ[nn])
        end
    
        a_N_grab_for_assign[:, tx] = wx_N[any_separate_list] - θₙ[any_separate_list]
    end

    y = transpose(2*find(x-> x>0, a_N_grab_for_assign) - 1)

    one_vs_all_sep_array_per = zeros(nr_Classes, 2)
    one_vs_all_sep_array_pe_top_n = zeros(nr_Classes, 2)

    samples_I_assign = zeros(1, nr_Classes)
    y_columns_sum_I = zeros(nr_Classes, any_separate_hp_total)

    samples_O_assign = zeros(1, nr_Classes)
    y_column_sums_O = zeros(nr_Classes, any_separate_hp_total)

    batch_range = 1:train_batch_size

    for dd in 1:nr_Classes
        samples_I_assign[dd] = count(x->x==Classes[dd], c1[batch_range])
        indexes_I = findall(x->x==Classes[dd], c1[batch_range])
        y_column_sums_I[dd, :] = sum(y[indexes_I, :], dims=1)
        samples_O_assign[dd] = count(x->x!=Classes[dd], c1[batch_range])
        indexes_O = findall(x->x!=Classes[dd], c1[batch_range])
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

#Estimate error
#Do same for Test Set:TODO