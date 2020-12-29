using LinearAlgebra

include("NeuronActivity.jl")

function NeuronLearningCycle(x::Array{Float64, 1}, w::Array{Float64,1}, θ::Array{Float64, 1}, μ₁::Array{Float64, 1}, μ₂::Array{Float64, 1}, c1::Array{Float64, 1}, c2::Array{Float64}, tₙ::Array{Float64, 1}, ϕ::Float64, ϵ::Float64, α::Float64, μᵉmode::Float64, μᵉpar::Float64, R_start::Int64, E_start::Int64)
    w_sqrt_norm = dot(transpose(w), w)
    w_norm = norm(w)
    wx = dot(w, x)

    #Rotation process
    if tₙ ≥ R_start
        #Define order of activation
        #Rotation conditioned that x is within ϕ from hyperplane
        if (wx ≥ (θ - ϕ)) & (wx ≤ (θ + ϕ)) 
             # 1 - Compute C, intersection point of segment (μ₂ - μ₁) within hyperplane 
             C = μ₁ + ((θ - dot(dot(transpose(w), μ₁), (μ₂ - μ₁)))/(dot(transpose(w), (μ₂ - μ₁))))
             # 2 - Compute E, intersection point of orthonormal projection of x into hyperplane 
             E = x + ((θₜ - wx)*w)./w_norm
             # 3 - Calculate vectors to define both planes, Po and P
             u = sign(wx - θ).*(w./w_norm)
             v = (E-C)./norm(E - C)
             # 4 - Calculate hyperplane (small) rotation, and small shift
             w = w + hcat(u, v) * [-(α^2)/2 -α; α -(α^2)/2] * [dot(w, u); dot(w, v)]
             w = w / norm(w)
             θ = dot(w, C)
        end
    end

    #Update Value
    wx = dot(w, x)
    #Shift process
    if (wx ≥ θ) & (wx ≤ (θ + ϕ))
        θ = θ - ϵ
        tₙ = tₙ + 1
    elseif (wx ≤ θ) & (wx ≥ (θ - ϕ))
        θ = θ + ϵ
        tₙ = tₙ + 1
    end

    # μ estimation process
    if tₙ ≥ E_start | tₙ <= E_start
        if μᵉmode == 0
            if (wx ≥ θ - μᵉpar) & (wx ≤ (θ + μᵉpar))
                if tₙ == E_start
                    if wx < θ
                        μ₂ = x - 2*abs(wx - θ)*w
                    else
                        μ₁ = x - 2*abs(wx - θ)*w
                    end

                    tₙ = tₙ + 1
                end
            
                p = 1
                if wx < θ
                    μ₁ = (c1 *μ₁ + p*x) / (c1 + p)
                    c1 = c1 + p
                else
                    μ₂ = (c2*μ₂ + p*x) / (c2 + p)
                    c2 = c2 + p
                end
        
            else
                p = 0
            end
        end
    end
    return w, θ, μ₁, μ₂, c1, c2, tₙ
end
