using MLDataUtils;
using Printf;

include("../Model/MultiEpoch.jl")
include("ErrFunction.jl")

X, Y = MLDataUtils.load_iris()
Xs, Ys = shuffleobs((X, Y))


for i in 1:10
    @time begin
        
        epoch_nr = 10000;
        nmr_training_batches = 1
        train_batch_size = 150
        d = 4
        hyp_nmr = 2
        Ω = 4.0
        σ = 0.8
        μᵉmode = 0.0
        μᵉpar = 6.4
        E_start = 100
        R_start = 250
        initial_orientation = "grid"

        (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(Xs, nmr_training_batches, d, train_batch_size, epoch_nr);
    end

    @printf "Model Accuracy: %.2f%%\n" get_model_acc(Xs, Ys, w_N, θ) * 100
end
