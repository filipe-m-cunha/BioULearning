using MLDatasets
using Images
using Printf

include("../Model/MultiEpoch.jl")
include("ErrFunction.jl")
include("../Gabor/GaborFilterDefiniton.jl")

train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()
X_prime = cat(train_x, test_x; dims = 3)
Y_train= vcat(train_y, test_y)[1:400]
Y_test = vcat(train_y, test_y)[401:500]
Xtemp = establishConnectionGabor(X_prime[:, :, 1:500], 20, 4, [4.6, 10.3], 4.8, [3.8, 5.] , 7. , 2.1, "winnerTakesAll")

X_t = zeros(196, 500)
for i in 1:500
    X_t[:, i] = reshape(Xtemp[i, :, :], 196, 1)
end

#=
X = X_t[:, 1:400]
X_test = X_t[:, 401:500]

#Fixed Parameters
epoch_nr = 3;
nmr_training_batches = 1
train_batch_size = 400
d = 196
hyp_nmr = 2
μᵉmode = 2.0
μᵉpar = 0.5
Ω = 4.0
ϵ = 0.005
α = 0.001
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

=#