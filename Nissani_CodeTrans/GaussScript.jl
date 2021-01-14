using Distributions, LinearAlgebra, Random, Gadfly
using DataFrames;
using CSV;
using HTTP;
using MLDataUtils;
using LinearAlgebra;
using JLD;

include("MultiEpochClassifier.jl")
include("MultiEpoch.jl")

ker1 = MvNormal(zeros(2), Diagonal(ones(2)))
ker2 = MvNormal([4.5; 4.5], Diagonal(ones(2)))

x1 = rand(ker1, 1000)
x2 = rand(ker2, 1000)

cv_X = shuffleobs(hcat(x1, x2))

@time begin
    epoch_nr = 100;
    nmr_training_batches = 1
    test_batch_size = 10
    train_batch_size = 2000
    classes = ["Ker1", "Ker2"]
    d = 2
    #hyp_nmr = 10
    (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = @fastmath MultiEpoch(cv_X, nmr_training_batches, d, train_batch_size, epoch_nr);
end

fs1(x) = (w_N[2,1]/(w_N[1,1] - θ[1]))*x + (w_N[1,2] - (w_N[2,1]/(w_N[1,1] - θ[1]))*w_N[1,1]);
fs2(x) = (w_N[2,2]/(w_N[1,2] - θ[2]))*x + (w_N[2,2] - (w_N[2,2]/(w_N[1,2] - θ[2]))*w_N[1,2]);
fs3(x) = (w_N[2,3]/(w_N[1,3] - θ[3]))*x + (w_N[1,3] - (w_N[2,3]/(w_N[1,3] - θ[3]))*w_N[1,3]);
fs4(x) = (w_N[2,4]/(w_N[1,4] - θ[4]))*x + (w_N[1,4] - (w_N[2,4]/(w_N[1,4] - θ[4]))*w_N[1,4]);
#fs5(x) = (w_N[2,5]/(w_N[1,5] - θ[5]))*x + (w_N[1,5] - (w_N[2,5]/(w_N[1,5] - θ[5]))*w_N[1,5]);
#fs6(x) = (w_N[2,6]/(w_N[1,6] - θ[6]))*x + (w_N[1,6] - (w_N[2,6]/(w_N[1,6] - θ[6]))*w_N[1,6]);
#fs7(x) = (w_N[2,7]/(w_N[1,7] - θ[7]))*x + (w_N[1,7] - (w_N[2,7]/(w_N[1,7] - θ[7]))*w_N[1,7]);
#fs8(x) = (w_N[2,8]/(w_N[1,8] - θ[8]))*x + (w_N[1,8] - (w_N[2,8]/(w_N[1,8] - θ[8]))*w_N[1,8]);

plot(
    layer(x=x1[1,:], y=x1[2,:], Geom.point, Theme(default_color=color("green"))),
    layer(x=x2[1,:], y=x2[2,:], Geom.point, Theme(default_color=color("blue"))),
    layer(fs1, -10, 10, Theme(default_color=color("red"))),
    layer(fs2, -10, 10, Theme(default_color=color("red"))),
    layer(fs3, -10, 10, Theme(default_color=color("red"))),
    layer(fs4, -10, 10, Theme(default_color=color("red")))
    #layer(fs5, -10, 10, Theme(default_color=color("red"))),
    #layer(fs6, -10, 10, Theme(default_color=color("red"))),
    #layer(fs7, -10, 10, Theme(default_color=color("red"))),
    #layer(fs8, -10, 10, Theme(default_color=color("red")))
)