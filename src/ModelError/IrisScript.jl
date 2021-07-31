using MLDataUtils;
using Printf;

include("../Model/MultiEpoch.jl")
include("ErrFunction.jl")

X, Y = MLDataUtils.load_iris()
Xs, Ys1 = shuffleobs((X, Y))
Ys = zeros(length(Ys1))
for i in 1:length(Ys)
    if(Ys1[i] == "setosa")
        Ys[i] = 1
    elseif(Ys1[i] == "versicolor")
        Ys[i] = 2
    elseif(Ys1[i] == "virginica")
        Ys[i] = 3

    end
end

#Fixed Parameters
epoch_nr = 100;
nmr_training_batches = 1
train_batch_size = 150
d = 4
hyp_nmr = 2
μᵉmode = 0.0
μᵉpar = 6.4

#=Parameters to evaluate
αval = [0.0005, 0.0001, 0.001, 0.0015, 0.0025, 0.005, 0.01]
ϵval = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.0025, 0.001]
ϕval = [0.5, 1.0, 2.0, 5.0, 10.0]
Ωval = [2.0, 4.0, 6.0, 8.0, 10.0]
σval = [0.5, 0.6, 0.7, 0.8, 1.0]
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
=#

epoch_nr = 100
nmr_training_batches = 1
train_batch_size = 150
d = 4
hyp_nmr = 4
μᵉmode = 0.0
μᵉpar = 6.4
Ω = 10.0
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
    (w_N, wx_N, θ, μ₁, μ₂, c1, c2, y_N) = MultiEpoch(Xs, Ys, Xs, Ys, nmr_training_batches, d, 
                                                    train_batch_size, epoch_nr,
                                                    hyp_nmr, Ω, ϵ, α, ϕ, σ, μᵉmode,
                                                    μᵉpar, E_start, Int(R_start*E_start),
                                                    initial_orientation, vary);
    println("Finished")
    acc, unlabeled, count = compAcc(Xs, Ys, Xs, Ys, w_N, θ, 3)
    @printf "Model Accuracy: %.2f%%\n" acc * 100
end