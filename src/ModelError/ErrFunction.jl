using MLDataUtils;
using LinearAlgebra;
using JLD;
using Printf;
using StatsBase;

#Given three input parameters, a point x, a Matrix
#wX and a vector θ, these last two describing the hyperplanes,
#the function will indicate relative location of the point x 
#according to each hyperplane

function placement(x, wX, θ)

    place = zeros(size(wX)[2], 1)

    for i in 1:size(wX)[2]

        place[i] = sign(transpose(wX[:, i]./sum(wX[:, i]))*x - θ[i])

    end

    return place
end

#Given three input parameters, a dataset X, a Matrix
#wX and a vector θ, these last two describing the hyperplanes,
#the function will return a matrix indicating the relative
#position of each point in the dataset according to each
#hyperplane

function placeDataset(X, wX, θ)

    fPlace = zeros(size(X)[2], size(wX)[2])

    for i in 1:size(X)[2]
        fPlace[i, :] = transpose(placement(X[:, i], wX, θ))
    end

    return fPlace
end

#Given a dataset X, a label set Y, a Matrix
#wX and a vector θ, these last two describing
#hyperplanes, the function will compute the
#accuracy of the model by comparing the
#labels of each point in the dataset with
#their respective relative hyperplane positions

function get_model_acc(X, Y, wX, θ)

    results = placeDataset(X, wX, θ)
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in 1:size(X)[2]

        for j in i:size(X)[2]

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

function label(X, Y, wX, θ, num)
    
    results = placeDataset(X, wX, θ)
    uniqueR = []
    labels = []
    count = 0
    for j in 1:length(results)
        if results[j] in uniqueR
            pos = findall(x-> x==results[j], uniqueR)
            if length(label[pos])==num
                label[pos] = int(mode(label[pos]))
            elseif  length(label[i])<num
                label[pos]+=[Y[j]]
            end
        else
            uniqueR += [results[j]]
            count += 1
        end
    end
    for j in labels
        labels[j] = mode(labels[j])
    end
    return uniqueR, labels, count
end

function compAcc(Xtrain, Ytrain, Xtest, Ytest, wX, θ, num)

    right = 0
    wrong = 0
    unlabeled = 0
    uniqueR, labels, count = label(Xtrain, Ytrain, wX, θ, num)
    results = placeDataset(Xtest, wX, θ)
    for j in 1:length(results)
        if results[j] in uniqueR
            pos = findall(x-> x==results[j], uniqueR)
            if labels[pos] == Ytest[j]
                right += 1
            else
                wrong += 1
            end
        else
            unlabeled += 1
        end
    end
    return (right/(right+wrong)), unlabeled, count
end

