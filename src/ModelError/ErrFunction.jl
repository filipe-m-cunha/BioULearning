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

function equalVec(vec1, vec2)
    if length(vec1) != length(vec2)
        return false
    else
        found = false
        i = 1
        while !found && i<length(vec1)
            if(vec1[i] != vec2[i])
                found = true
            end
            i = i+1
        end
        return !found
    end
end

function findRow(data, row)
    if(data == Any[])
        return -1
    else
        i = 1
        found = false
        while i<=length(data) && !found
            if(equalVec(data[i], row))
                found = true
            else
                i = i+1
            end
        end
        if found
            return i
        else
            return -1
        end
    end
end


function label(X, Y, counter)

    results = X
    y = Any[]
    z = Float64[]
    unique = Any[]
    for i in 1:size(results)[1]
        j = findRow(unique, results[i, :])
        if(j == -1)
            push!(unique, results[i, :])
            push!(z, 1)
            yval = zeros(counter)
            yval[1] = Y[i]
            push!(y, yval)
        else
            if(z[j] < counter)
                z[j] = z[j] + 1
                y[i][z[j]] = Y[i]
            end
        end
    end
    retY = zeros(size(results)[1])
    uncertain = 0
    for k in 1:size(results)[1]
        k1 = findRow(unique, results[k, :])
        if( k1 == -1)
            println("Error: Something wrong")
            break
        else
            k2 = convert(Int64, z[k1])
            if(k2 >= counter)
                retY[k] = convert(Int64, mode(y[k1]))
            else
                retY[k] = -1
                uncertain = uncertain + 1
            end
        end
    end
    return retY, uncertain, unique
end

x = [1 -1 1; -1 1 -1; 1 -1 1; -1 1 -1; 1 1 1; -1 1 -1]
y = [1; 2; 1; 2; 3; 1]
retY, uncertain = label(x, y, 1)
