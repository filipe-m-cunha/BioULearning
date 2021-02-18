using MLDataUtils;
using Statistics;
using LIBSVM;

#Loading the datasets

X, Y = MLDataUtils.load_iris()
Xs, Ys = shuffleobs((X, Y))
Ysi = zeros(size(Ys))
for ys in 1:size(Ys)[1]
    if Ys[ys] == "virginica"
        Ysi[ys] = 1
    elseif Ys[ys] == "setosa"
        Ysi[ys] = 2
    else
        Ysi[ys] = 3
    end
end

#Train-test split

((train_X, train_Y), (test_X, test_Y)) = splitobs((Xs, Ysi); at = 0.85)

#Just computes the euclidian distance between two points
function euclidianDistance(x, y)
    return dot(x-y, x-y)
end 

#Given a datapoint and a set x, will find the nearest k neighbours of x
function get_k_nearest_neighbours(x, x_test, i, k)

    nRows, nCols = size(x)
    imageI = zeros(nRows)
    imageJ = zeros(nRows)

    for index in 1:nRows

        imageI[index] = x_test[index, i]

    end

    distances = zeros(nCols)

    for j in 1:nCols
        for index in 1:nRows

            imageJ[index] = x[index, j]

        end

        distances[j] = euclidianDistance(imageI, imageJ)

    end

    sortedNeighbours = sortperm(distances)
    kNearestNeighbours = sortedNeighbours[2:k+1]

    return kNearestNeighbours
end

#Function that assigns the labels to each point in a dataset

function assign_labels(x, x_test, y, k, i)

    kNearestNeighbours = get_k_nearest_neighbours(x, x_test, i, k)
    nearest_labels = y[kNearestNeighbours]

    return round(mean(nearest_labels))
end

#Compute the accuracy and training time of the knn model for several k values

for k in 1:15

    @time begin

        yPredictionsk = [assign_labels(train_X, test_X, train_Y, k, i) for i in 1:size(test_X, 2)]
        accuracyk = mean(yPredictionsk .== test_Y)
        println("The loofCV accuracy of $(k)NN is $(accuracyk)")

    end
end

#Compute the accuracy and training time of the SVM-model

@time begin
    model = svmtrain(train_X, train_Y)
    ŷ, decision_values = svmpredict(model, test_X)
    @printf "SVM Accuracy: %.2f%%\n" mean(ŷ .== test_Y) * 100
end
