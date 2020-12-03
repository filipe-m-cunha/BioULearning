#= Inputs (Hul learning function):
x  -- Input vector
w  -- Neural Weight vector
θ -- Activity threshold
μ₁ᵉ, μ₂ᵉ -- Current Networks hp 2 sides mean estimates
c1, c2 -- Current Neurons hp 2 sides means estimate cumulative weighting factor

t_n -- neuron current self-time

Input (Other)

ϕ
ϵ
a 
mu_est_mode 
mu_est_par 
R_start 
E_start 
d 

Output:
w -- updated Weight vector
theta -- updated activity threshold
mul1_est, mu2_est -- Updated hp 2 sides mean estimates
c1, c2 -- Updated hp 2 sides mean estimates cumulative weighting factor
t_N -- updated neuron current time
=# 

function NeuronLearningCycle(x, w, μ₁ᵉ, μ₂ᵉ, c1, c2, tₙ)
    w_sqrt_norm = transpose(w).*w
    w_norm = norm(w)
    wx = transpose(w) .* x

    #Rotation process
    if tₙ ≥ R_start
        #Define order of activation
        #Rotation conditioned that x is within ϕ from hyperplane
        if (wx ≥ (θ - ϕ)) & (wx ≤ (θ + ϕ))
             # 1 - Compute C, intersection point of segment (μ₂ - μ₁) within hyperplane 
             C = μ₁ᵉ + ((θ - transpose(w)*μ₁ᵉ)/(transpose(w) * (μ₂ᵉ - μ₁ᵉ)))*(μ₂ᵉ - μ₁ᵉ)
             # 2 - Compute E, intersection point of orthonormal projection of x into hyperplane 
             E = x + ((θ - transpose(w)*x)/w_sqrt_norm)*w
             # 3 - Calculate vectors to define both planes, Po and P
             u = (sign(transpose(w)*x - θ)/w_norm)*w
             v = (E-C) / sqrt(transpose(E-C) * (E-C))
             # 4 - Calculate hyperplane (small) rotation, and small shift
             w = w + hcat(u, v) * vcat(hcat(-a_sq, -a), hcat(a, -a_sq)) * vcat(transpose(w)*u, transpose(w)*v)
             w = w / norm(w)
             θ = transpose(w) * C
        end
    end

    wx = transpose(w)*x
    #Shift process
    if (wx ≥ θ) & (wx ≤ (θ + ϕ))
        θ = θ - ϵ
        tₙ = tₙ + 1
    end

    # μ estimation process
    if tₙ ≥ E_start
        if μᵉmode == 0
            if (wx ≥ θ - μᵉpar) & (wx ≤ (θ + μᵉpar))
                if tₙ == E_start
                    if wx < θ
                        μ₂ᵉ = x - 2*abs(wx - θ) * w
                    else
                        μ₁ᵉ = x - 2*abs(wx - θ)*w
                    end

                    tₙ = tₙ + 1
                end
            
            p = 1
            if wx < θ
                μ₁ᵉ = (c1 *μ₁ᵉ + p*x) / (c1 + p)
                c1 = c1 + p
            else
                μ₂ᵉ = (c2*μ₂ᵉ + p*x) / (c2 + p)
                c2 = c2 + p
            end
        
        else
            p = 0
        end
    end
end