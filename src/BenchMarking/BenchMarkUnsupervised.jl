using Printf;
using Clustering;
using MLDataUtils; 
using Plots;
using Statistics;

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



@time begin
    result = kmeans(Xs, 3);
end

@printf "K-Means Accuracy: %.2f%%\n" mean(Ysi .== result.assignments) * 100