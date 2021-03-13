using MLDatasets
using Images
using Printf

include("../Model/MultiEpoch.jl")
include("ErrFunction.jl")

train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

X_prime = train_x[:, :, 1:1000];
Y= train_y[1:1000];

X = zeros(784, 1000)
for i in 1:1000
    X[:, i] = reshape(X_prime[:, :, i]./255, 784, 1)
end

#Fixed Parameters
epoch_nr = 10;
nmr_training_batches = 1
train_batch_size = 1000
d = 784
hyp_nmr = 2
μᵉmode = 0.0
μᵉpar = 4.0
Ω = 1.0
ϵ = 0.0005
α = 0.001
σ = 0.8
ϕ = 2.0
E_start = 200
R_start = 1.5
initial_orientation = "random"
vary = 0

@time begin
    println("Starting up...")
    (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(X, nmr_training_batches, d, 
                                                    train_batch_size, epoch_nr,
                                                    hyp_nmr, Ω, ϵ, α, ϕ, σ, μᵉmode,
                                                    μᵉpar, E_start, Int(R_start*E_start),
                                                    initial_orientation, vary);
    println("Finished")
    acc = get_model_acc(X, Y, w_N, θ)
    @printf "Model Accuracy: %.2f%%\n" acc * 100
end