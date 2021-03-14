using MLDatasets
using Images
using Printf

include("../Model/MultiEpoch.jl")
include("ErrFunction.jl")
include("../Gabor/GaborFilterDefiniton.jl")

train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

X_prime = cat(train_x, test_x; dims = 3);
Y= vcat(train_y, test_y);

Xtemp = establishConnectionGabor(X_prime, 4, 4, [4.6, 10.3], 4.8, [3.8, 5.] , 7. , 2.1, "winnerTakesAll")

X = zeros(196, 70000)
for i in 1:70000
    X[:, i] = reshape(Xtemp[i, :, :], 196, 1)
end

#Fixed Parameters
epoch_nr = 10;
nmr_training_batches = 1
train_batch_size = 1000
d = 196
hyp_nmr = 2
μᵉmode = 0.0
μᵉpar = 4.0
Ω = 5.0
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
    (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(X[:, 1:1000], X, Y, nmr_training_batches, d, 
                                                    train_batch_size, epoch_nr,
                                                    hyp_nmr, Ω, ϵ, α, ϕ, σ, μᵉmode,
                                                    μᵉpar, E_start, Int(R_start*E_start),
                                                    initial_orientation, vary);
    println("Finished")
    acc = get_model_acc(X, Y, w_N, θ)
    @printf "Model Accuracy: %.2f%%\n" acc * 100
end