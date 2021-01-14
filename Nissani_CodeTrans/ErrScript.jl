using DataFrames;
using CSV;
using HTTP;
using MLDataUtils;
using LinearAlgebra;
using JLD;

include("MultiEpoch.jl")
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
    epoch_nr = 10000;
    nmr_training_batches = 1
    test_batch_size = 10
    train_batch_size = 128
    classes = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
    d = 4
    #hyp_nmr = 10
    (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = @fastmath MultiEpoch(cv_X, nmr_training_batches, d, train_batch_size, epoch_nr);
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
