using MLDatasets
using Images
using Printf
using Random

include("../Model/MultiEpoch.jl")
include("ErrFunction.jl")
include("../Gabor/GaborFilterDefiniton.jl")

Random.seed!(200)

#=
train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()
X_prime = cat(train_x, test_x; dims = 3)
Y_train= vcat(train_y, test_y)[1:4000]
Y_test = vcat(train_y, test_y)[4001:5000]
Xtemp, gaborBank = establishConnectionGabor(X_prime[:, :, 1:5000], 20, 4, [4.6, 10.3], 4.8, [3.8, 5.] , 7. , 2.1, "winnerTakesAll")


X_t = zeros(625, 5000)
for i in 1:5000
    X_t[:, i] = reshape(Xtemp[i, :, :], 625, 1)
end


=#


X = X_t[:, 1:400]
X_test = X_t[:, 401:500]



#Fixed Parameters
epoch_nr = 50;
nmr_training_batches = 1
train_batch_size = 400
d = 625
hyp_nmr = 4
μᵉmode = 2.0
μᵉpar = 0.5
Ω = 4.0
ϵ = 0.0005
α = 0.00001
σ = 0.8
ϕ = 2.0
E_start = 200
R_start = 1.5
initial_orientation = "random"
vary = 0

@time begin
    println("Starting up...")
    (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(X, Y_train, X_test, Y_test, nmr_training_batches, d, 
                                                    train_batch_size, epoch_nr,
                                                    hyp_nmr, Ω, ϵ, α, ϕ, σ, μᵉmode,
                                                    μᵉpar, E_start, Int(R_start*E_start),
                                                    initial_orientation, vary);
    println("Finished")
    acc = get_model_acc(X_test, Y_test, w_N, θ)
    @printf "Model Accuracy: %.2f%%\n" acc * 100
end

