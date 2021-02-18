using Distributions;
using LinearAlgebra;
using Random;
using Gadfly;
using Interact;
using DataFrames;
using CSV;
using HTTP;
using MLDataUtils;
using LinearAlgebra;
using JLD;
using Cairo;

include("../Model/MultiEpoch.jl")

#=Requires notebook to be run (Gulbenkian.ijulia)
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
=#
#If jupyterlab not installed, run (just for xn=yn=-2.5):
#Create the Gaussian distributions

ker1 = MvNormal([-2.; -2.], Diagonal([1; 1]))
#ker2 = MvNormal([6.5; 6.5], Diagonal(ones(2)))
ker3 = MvNormal([-3.5, 3.5], Diagonal([1; 1]))
#Create the kernels to be evaluated
x1 = rand(ker1, 1000)
#x2 = rand(ker2, 1000)
x3 = rand(ker3, 1000)
#Shuffle the kernels
cv_X = shuffleobs(hcat(x1, x3))
#Initialize hyperparameters
epoch_nr = 100000;
nmr_training_batches = 1
test_batch_size = 10
train_batch_size = 2000
classes = ["Ker1", "Ker3"]
d = 2
#Train the model
(w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(cv_X, nmr_training_batches, d, train_batch_size, epoch_nr)
#=Get equations of lines defining the hyperplanes
θshift = 1
nmr_hyp=8
Ω = -0.5
σ = 0.8
θspacing = Ω/(nmr_hyp + 1)
ϵ = 2*σ
α = 0.5
ϕ = 2.0
nr_neurons = 8*nmr_hyp
w_N = zeros(d, nr_neurons)
for i in 1:d:nr_neurons-1
    w_N[:, i:(i+d-1)] = Diagonal(ones(d))
end
#println(size(w_N))
θ = θspacing*ones(d, nmr_hyp)
for i in 1:nmr_hyp
    θ[:, i] = i*θ[:, i]
end
θ = reshape(θ, size(θ)[1]*size(θ)[2], 1)
θ = θ .+ θshift
=#
fs1(x) = (w_N[2,1]/(w_N[1,1] - θ[1]))*x + (w_N[1,2] - (w_N[2,1]/(w_N[1,1] - θ[1]))*w_N[1,1]);
fs2(x) = (w_N[2,2]/(w_N[1,2] - θ[2]))*x + (w_N[2,2] - (w_N[2,2]/(w_N[1,2] - θ[2]))*w_N[1,2]);
fs3(x) = (w_N[2,3]/(w_N[1,3] - θ[3]))*x + (w_N[1,3] - (w_N[2,3]/(w_N[1,3] - θ[3]))*w_N[1,3]);
fs4(x) = (w_N[2,4]/(w_N[1,4] - θ[4]))*x + (w_N[1,4] - (w_N[2,4]/(w_N[1,4] - θ[4]))*w_N[1,4]);
fs5(x) = (w_N[2,5]/(w_N[1,5] - θ[5]))*x + (w_N[1,5] - (w_N[2,5]/(w_N[1,5] - θ[5]))*w_N[1,5]);
fs6(x) = (w_N[2,6]/(w_N[1,6] - θ[6]))*x + (w_N[1,6] - (w_N[2,6]/(w_N[1,6] - θ[6]))*w_N[1,6]);
fs7(x) = (w_N[2,7]/(w_N[1,7] - θ[7]))*x + (w_N[1,7] - (w_N[2,7]/(w_N[1,7] - θ[7]))*w_N[1,7]);
fs8(x) = (w_N[2,8]/(w_N[1,8] - θ[8]))*x + (w_N[1,8] - (w_N[2,8]/(w_N[1,8] - θ[8]))*w_N[1,8]);
fs9(x) = (w_N[2,9]/(w_N[1,9] - θ[9]))*x + (w_N[1,9] - (w_N[2,9]/(w_N[1,9] - θ[9]))*w_N[1,9]);
fs10(x) = (w_N[2,10]/(w_N[1,10] - θ[10]))*x + (w_N[1,10] - (w_N[2,10]/(w_N[1,10] - θ[10]))*w_N[1,10]);
fs11(x) = (w_N[2,11]/(w_N[1,11] - θ[11]))*x + (w_N[1,11] - (w_N[2,11]/(w_N[1,11] - θ[11]))*w_N[1,11]);
fs12(x) = (w_N[2,12]/(w_N[1,12] - θ[12]))*x + (w_N[1,12] - (w_N[2,12]/(w_N[1,12] - θ[12]))*w_N[1,12]);
fs13(x) = (w_N[2,13]/(w_N[1,13] - θ[13]))*x + (w_N[1,13] - (w_N[2,13]/(w_N[1,13] - θ[13]))*w_N[1,13]);
fs14(x) = (w_N[2,14]/(w_N[1,14] - θ[14]))*x + (w_N[1,14] - (w_N[2,14]/(w_N[1,14] - θ[14]))*w_N[1,14]);
fs15(x) = (w_N[2,15]/(w_N[1,15] - θ[15]))*x + (w_N[1,15] - (w_N[2,15]/(w_N[1,15] - θ[15]))*w_N[1,15]);
fs16(x) = (w_N[2,16]/(w_N[1,16] - θ[16]))*x + (w_N[1,16] - (w_N[2,16]/(w_N[1,16] - θ[16]))*w_N[1,16]);
#Plot the required graphics
plot(
    layer(x=x1[1,:], y=x1[2,:], Geom.point, Theme(default_color=color("green"))),
    #layer(x=x2[1,:], y=x2[2,:], Geom.point, Theme(default_color=color("blue"))),
    layer(x=x3[1,:], y=x3[2,:], Geom.point, Theme(default_color=color("orange"))),
    layer(fs1, -10, 10, Theme(default_color=color("red"))),
    layer(fs2, -10, 10, Theme(default_color=color("red"))),
    layer(fs3, -10, 10, Theme(default_color=color("red"))),
    layer(fs4, -10, 10, Theme(default_color=color("red"))),
    layer(fs5, -10, 10, Theme(default_color=color("red"))),
    layer(fs6, -10, 10, Theme(default_color=color("red"))),
    layer(fs7, -10, 10, Theme(default_color=color("red"))),
    layer(fs8, -10, 10, Theme(default_color=color("red"))),
    layer(fs9, -10, 10, Theme(default_color=color("red"))),
    layer(fs10, -10, 10, Theme(default_color=color("red"))),
    layer(fs11, -10, 10, Theme(default_color=color("red"))),
    layer(fs12, -10, 10, Theme(default_color=color("red"))),
    layer(fs13, -10, 10, Theme(default_color=color("red"))),
    layer(fs14, -10, 10, Theme(default_color=color("red"))),
    layer(fs15, -10, 10, Theme(default_color=color("red"))),
    layer(fs16, -10, 10, Theme(default_color=color("red")))
)

#If saving image locally is required, uncomment following line
#draw(PNG("plot10.png"), ploting)