using Distributions
using LinearAlgebra
using Random
using Gadfly
using Interact
using DataFrames;
using CSV;
using HTTP;
using MLDataUtils;
using LinearAlgebra;
using JLD;

include("MultiEpoch.jl")
include("MultiEpoch.jl")

#Requires notebook to be run (Gulbenkian.ijulia)
@manipulate for xn = -10:0.1:1, yn = -10:0.1:1

    #Create the Gaussian distributions
    ker1 = MvNormal([xn;yn], Diagonal(ones(2)))
    ker2 = MvNormal([4.5; 4.5], Diagonal(ones(2)))

    #Create the kernels to be evaluated
    x1 = rand(ker1, 1000)
    x2 = rand(ker2, 1000)

    #Shuffle the kernels
    cv_X = shuffleobs(hcat(x1, x2))

    #Initialize hyperparameters
    epoch_nr = 100;
    nmr_training_batches = 1
    test_batch_size = 10
    train_batch_size = 2000
    classes = ["Ker1", "Ker2"]
    d = 2
    
    #Train the model
    (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(cv_X, nmr_training_batches, d, train_batch_size, epoch_nr);

    #Get equations of lines defining the hyperplanes
    fs1(x) = (w_N[2,1]/(w_N[1,1] - θ[1]))*x + (w_N[1,2] - (w_N[2,1]/(w_N[1,1] - θ[1]))*w_N[1,1]);
    fs2(x) = (w_N[2,2]/(w_N[1,2] - θ[2]))*x + (w_N[2,2] - (w_N[2,2]/(w_N[1,2] - θ[2]))*w_N[1,2]);
    fs3(x) = (w_N[2,3]/(w_N[1,3] - θ[3]))*x + (w_N[1,3] - (w_N[2,3]/(w_N[1,3] - θ[3]))*w_N[1,3]);
    fs4(x) = (w_N[2,4]/(w_N[1,4] - θ[4]))*x + (w_N[1,4] - (w_N[2,4]/(w_N[1,4] - θ[4]))*w_N[1,4]);
    #fs5(x) = (w_N[2,5]/(w_N[1,5] - θ[5]))*x + (w_N[1,5] - (w_N[2,5]/(w_N[1,5] - θ[5]))*w_N[1,5]);
    #fs6(x) = (w_N[2,6]/(w_N[1,6] - θ[6]))*x + (w_N[1,6] - (w_N[2,6]/(w_N[1,6] - θ[6]))*w_N[1,6]);
    #fs7(x) = (w_N[2,7]/(w_N[1,7] - θ[7]))*x + (w_N[1,7] - (w_N[2,7]/(w_N[1,7] - θ[7]))*w_N[1,7]);
    #fs8(x) = (w_N[2,8]/(w_N[1,8] - θ[8]))*x + (w_N[1,8] - (w_N[2,8]/(w_N[1,8] - θ[8]))*w_N[1,8]);

    #Plot the required graphics
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
end