using DataFrames;
using CSV;
using HTTP;
using MLDataUtils;
using LinearAlgebra;
using JLD;
using Printf;

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

function placement(x, wX, θ)
    place = zeros(size(wX)[2], 1)
    for i in 1:size(wX)[2]
        place[i] = sign(transpose(wX[:, i]./sum(wX[:, i]))*x - θ[i])
    end
    return place
end

function placeDataset(X, wX, θ)
    fPlace = zeros(size(X)[2], size(wX)[2])
    for i in 1:size(X)[2]
        fPlace[i, :] = transpose(placement(X[:, i], wX, θ))
    end
    return fPlace
end

function get_model_acc(X, Y, wX, θ)

    results = placeDataset(X, wX, θ)
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in 1:size(X)[2]
        for j in 1:size(X)[2]
            if results[i, :] == results[j, :]
                if Y[i] == Y[j]
                    true_positive += 1
                else
                    false_positive += 1
                end
            else
                if Y[i] == Y[j]
                    false_negative += 1
                else
                    true_negative += 1
                end
            end
        end
    end
    accuracy = (true_positive + true_negative)/(true_negative + true_positive + false_negative + false_positive)

    return accuracy
end
for i in 1:10
    @time begin
        epoch_nr = 10000;
        nmr_training_batches = 1
        test_batch_size = 10
        train_batch_size = 128
        classes = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"]
        d = 4
        #hyp_nmr = 10
        (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(Xs, nmr_training_batches, d, train_batch_size, epoch_nr);
    end
    @printf "Model Accuracy: %.2f%%\n" get_model_acc(Xs, Ys, w_N, θ) * 100
end
