using MLDatasets
using Images
using Printf
using Random

include("../Model/MultiEpoch.jl")
include("ErrFunction.jl")
include("../Gabor/GaborFilterDefiniton.jl")


for i in 1:10
Random.seed!(i*221)


train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()
X_prime = cat(train_x, test_x; dims = 3)
Y_train= vcat(train_y, test_y)[1:6000]
Y_test = vcat(train_y, test_y)[6001:7000]
Xtemp, gaborBank = establishConnectionGaborAlt(X_prime[:, :, 1:7000], 20, 4, [4.6, 10.3], 4.8, [3.8, 5.] , 7. , 2.1, "winnerTakesAll", 2, 2)


X_t = zeros(784, 7000)
for i in 1:7000
    X_t[:, i] = reshape(X_prime[:, :, i], 784, 1)
end





X = X_t[:, 1:6000]
X_test = X_t[:, 6001:7000]



#Fixed Parameters
epoch_nr = 10;
nmr_training_batches = 1
train_batch_size = 6000
d = 784
hyp_nmr = 4
μᵉmode = 2.0
μᵉpar = 0.5
Ω = 25.0
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
    #acc = get_model_acc(X_test, Y_test, w_N, θ)
    #@printf "Model Accuracy: %.2f%%\n" acc * 100
end
println("seed: ")
println(i*221)
end 
