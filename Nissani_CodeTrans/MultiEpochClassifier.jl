using DataFrames
using CSV
using HTTP
using MLDataUtils

data_link = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = DataFrame!(CSV.File(HTTP.get(data_link).body; header = false))


X, Y = MLDataUtils.load_iris()
Xs, Ys = shuffleobs((X, Y))

((cv_X, cv_Y), (test_X, test_Y)) = splitobs((Xs, Ys); at = 0.85)

@time begin
    start_from_first_batch = 1

    if start_from_first_batch == 1
        classifier_features_file = Xs

        epoch_nr = 1
        nmr_training_batches = 10
        test_batch_size = 10
        classes = ["Iris-setosa", "Iris-virginica"]
        d = 4
        train_batch_size = 10
        nr_Classes = 2

        if display_class_names == 1
            println(nr_Classes, "Classes, ")
            for i in 1:nr_Classes
                println(classes[i])
            end
        end

        hyp_nmr = 2 #Change according to nmr of hyperplanes per dim
        Ω = 2.5
        θshift = 0
        θspacing = Ω/(hyp_nmr + 1)

        σ = 0.8
        ϵ = 0.0033*σ
        α = 0.04
        Φ = 2*σ

        μᵉmode = 0
        μᵉpar = 4

        E_start = 150
        Time_vary = 0

        nr_neurons = d*hyp_nmr

        w_N = zeros(d, nr_neurons)

        for i in 1:d:nr_neurons-1
            w_N[:, i:(i+d-1)] = I(d)
        end

        θₙ = θspacing*ones(d, hyp_nmr)

        c1 = zeros(nr_neurons, 1)
        c2 = zeros(nr_neurons, 1)

        μ₁ = zeros(d, nr_neurons)
        μ₂ = zeros(d, nr_neurons)

        a_N = zeros(nr_neurons, 1)
        y_N = zeros(nr_neurons, 1)

        wx_N = zeros(nr_neurons, 1)
        tₙ = zeros(nr_neurons, 1)

        ss = 0
        b_start = 1

    end

    for j in 1:nmr_training_batches*epoch_nr
        if time_var ==1
            ϵ = ϵ*ϵvar^(j-1)
            α = α*αvat^(j-1)
            Φ = Φ*Φvar^(j-1)
        end

        for k in 1:nr_neurons
            ss = ss + 1

            (w_N[:, k], θₜ[k], μ₁[:, k], μ₂[:, k], c1[k], c2[k], tₙ[k]) = NeuronLearningCycle(x,w_N[:, k], θₜ[k], μ₁[:, k], μ₂[:, k], c1[k], c2[k], tₙ[k], Φ, ϵ, α,  μᵉmode, μᵉpar, R_start, E_start)
            (y_N[k], wx_N[k]) = NeuronActivity(x, w_N[:, k], θₙ[k])
        end
    end
end