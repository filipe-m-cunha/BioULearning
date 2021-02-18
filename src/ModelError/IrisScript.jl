using MLDataUtils;
using Printf;

include("../Model/MultiEpoch.jl")
include("ErrFunction.jl")

X, Y = MLDataUtils.load_iris()
Xs, Ys = shuffleobs((X, Y))

#Fixed Parameters
epoch_nr = 100;
nmr_training_batches = 1
train_batch_size = 150
d = 4
hyp_nmr = 2
μᵉmode = 0.0
μᵉpar = 6.4

#Parameters to evaluate
αval = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
ϵval = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.001]
ϕval = [0.5, 1.0, 2.0, 5.0, 10.0]
Ωval = [2.0, 4.0, 6.0, 8.0, 10.0]
σval = [0.5, 0.8, 1]
E_startval = [100, 160, 300]
R_startval = [0.75, 1, 1.5]
initial_orientationval = ["random", "grid"]
varyval = [0, 1]
best = 0.0
best_val = zeros(9)
#Run model for the parameters to evaluate
for α in αval
    println("Alpha at: ", α)
    for ϵ in ϵval
        println("Epsilon at: ", ϵ)
        for ϕ in ϕval
            for Ω in Ωval
                for σ in σval
                    for E_start in E_startval
                        for R_start in R_startval
                            for initial_orientation in initial_orientationval
                                for vary in varyval
                                    (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(Xs, nmr_training_batches, d, 
                                                                                    train_batch_size, epoch_nr,
                                                                                    hyp_nmr, Ω, ϵ, α, ϕ, σ, μᵉmode,
                                                                                    μᵉpar, E_start, Int(R_start*E_start),
                                                                                    initial_orientation, vary);
                                    acc = get_model_acc(Xs, Ys, w_N, θ)
                                    if acc > best
                                        global best = acc
                                        global best_val = [α, ϵ, ϕ, Ω, σ, E_start, E_start*R_start, initial_orientation, vary]
                                        @printf "Model Accuracy (New Bes): %.2f%%\n" acc * 100
                                        println("Alpha: ", α, " Epsilon: ", ϵ, " Phi: ", ϕ, " Omega: ", Ω,
                                                " Sigma: ", σ, " E_Start: ", E_start, " R_Start: ", R_start*E_start,
                                                " Orientation: ", initial_orientation, " Vary: ", vary)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
