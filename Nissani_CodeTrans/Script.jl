using DataFrames;
using CSV;
using HTTP;
using MLDataUtils;
using LinearAlgebra;

include("MultiEpochClassifier.jl")
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
    w_N, θₙ, μ₁, μ₂, c1, c2, tₙ, y_N, wx_N = MultiEpochClassifier(cv_X, nmr_training_batches, 
                                                                classes, d, train_batch_size, epoch_nr,
                                                                display_class_names, hyp_nmr, Ω, θshift, σ, μᵉmode, 
                                                                μᵉpar, E_start, R_start, time_var, Φvar);
end

y_N = zeros(nr_neurons, 1);
wx_N = zeros(nr_neurons, 1)

separateMatrix = zeros(nr_Classes, nr_neurons)

for nn = 1:nr_neurons
    for dd = 1:nr_Classes
        do(nothing)
    end
end
    