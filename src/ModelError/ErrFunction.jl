using MLDataUtils;
using LinearAlgebra;
using JLD;
using Printf;

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