using DataFrames;
using CSV;
using HTTP;
using MLDataUtils;
using LinearAlgebra;
using JLD;

include("MultiEpochClassifier.jl")
#include("RG_Unsuper\GLBNK-Unsupervised\helpers\helpers.jl")

data_link = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = DataFrame!(CSV.File(HTTP.get(data_link).body; header = false))

X, Y = MLDataUtils.load_iris()
Xs, Ys = shuffleobs((X, Y))

((cv_X, cv_Y), (test_X, test_Y)) = splitobs((Xs, Ys); at = 0.85)

function softmax(a)
    c = maximum(a)
    exp.(a .- c) / sum(exp.(a .- c))
end

@time begin
    epoch_nr = 10;
    nmr_training_batches = 1
    test_batch_size = 10
    train_batch_size = 128
    classes = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
    d = 4; 
    #display_class_names = 0;
    #hyp_nmr = 2
    #Ω = 2.5
    #θshift = 0.0
    #σ = 0.8
    #μᵉmode = 0.0
    #μᵉpar = 4.0
    #E_start = 15
    #R_start = 200
    #time_var = 1
    #Φvar = 1.0
    w_N, θ, μ₁, μ₂, c1, c2, tₙ, y_N, wx_N, rotC, ShiftLC, ShiftRC = MultiEpochClassifier(cv_X, nmr_training_batches, 
                                                                classes, d, train_batch_size, epoch_nr);
end

function placement(x, wX)
    place = zeros(size(wX)[2], 1)
    for i in 1:size(wX)[2]
        place[i] = sign(transpose(wX[:, i])*x)
    end
    return place
end

function placeDataset(X, wX)
    fPlace = zeros(size(X)[2], size(wX)[2])
    for i in 1:size(X)[2]
        fPlace[i, :] = transpose(placement(X[:, i], wX))
    end
    return fPlace
end