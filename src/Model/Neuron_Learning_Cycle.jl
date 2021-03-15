using LinearAlgebra

include("NeuronActivity.jl")

#Represents the Learning Cycle for a single neuron. The function should take as inputs a vector X (a line from the analysed datset),
#a vector w (describing the hyperplane to be shifted/rotated), a value θ, the averages of the two classes that the hyperplane differentiates
#(μ₁ and μ₂), the points c1 and c2 (defining the rotation), the timer tₙ, and the hyperparameters ϕ, ϵ, α, μᵉmode, μᵉpar, R_start, E_start.
#The function will then proceed to rotate or shift the hyperplane if it meets the required constraints.
function NeuronLearningCycle(x::Array{Float64, 1}, w::Array{Float64,1}, θ::Float64, 
                            μ₁::Array{Float64, 1}, μ₂::Array{Float64, 1}, c1,
                            c2, tₙ, ϕ::Float64, ϵ::Float64, α::Float64, 
                            μᵉmode::Float64, μᵉpar::Float64, R_start::Int64, E_start::Int64) #, rotC::Int64, ShiftLC::Int64, ShiftRC::Int64)

    #Calculate norm, squared norm and dot product between w and x, in order to store the values.
    w_sqrt_norm = transpose(w)*w
    w_norm = norm(w)
    wx = transpose(w)*x
    #Rotation process
    if tₙ ≥ R_start
        #println("Here")
        #Rotation will only be applied if x is sufficiently close (distance ϕ) to the hyperplane.
        if (wx ≥ (θ - ϕ)) & (wx ≤ (θ + ϕ))
            #rotC += 1
            # 1 - Compute C, intersection point of segment (μ₂ - μ₁) with the hyperplane 
            C = μ₁ + ((θ - transpose(w)*μ₁)/(transpose(w)*(μ₂ - μ₁)))*(μ₂ - μ₁)
            # 2 - Compute E, intersection point of orthonormal projection of x into hyperplane 
            E = x + ((θ - transpose(w)*x)/w_sqrt_norm)*w
            # 3 - Calculate vectors (u, v) to define both auxiliary planes
            u = (sign(transpose(w)*x - θ)/w_norm)*w
            v = (E-C)/sqrt(transpose(E-C)*(E-C))
            # 4 - Apply the rotation to the hypeplane
            w = w + hcat(u, v) * [-(α^2)/2  -α; α  -(α^2)/2] * [transpose(w)*u; transpose(w)*v]
            w = w / norm(w)
            θ = transpose(w)*C
        end
    end

    #Update value of the dot product between w and x
    #println("w: ", w)
    wx = transpose(w)*x
    #println("wx: ", wx)
    #println("x: ", x)
    #Shift process
    #If x is near hyperplane (at a distance ϕ) and on the right, shift hyperplane to the left.
    if θ < wx ≤ θ + ϕ
        #ShiftRC += 1
        θ = θ - ϵ
        tₙ = tₙ + 1
    end
    #If x is near hyperplane (at a distance ϕ) and on the left, shift hyperplane to the right.
    if θ - ϕ ≤ wx ≤ θ
        #ShiftLC += 1
        θ = θ + ϵ
        tₙ = tₙ + 1

    end

    # μ estimation process
    if tₙ ≥ E_start
        if μᵉmode == 0 #Assumption for the model to work properly
            #If wx is near θ (distance μᵉpar), change means accordingly
            if (wx ≥ θ - μᵉpar) & (wx ≤ (θ + μᵉpar))
                if tₙ == E_start
                    #println("Here1")
                    #If sample is in original half-space
                    if wx < θ
                        μ₂ = x + 2*abs(wx - θ)*w
                    #If sample is not in original half-space
                    else
                        μ₁ = x - 2*abs(wx - θ)*w
                    end
                    #Update hyperplane self timer
                    tₙ = tₙ + 1
                end
                
                #Calculation of the weight function update
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
    return w, θ, μ₁, μ₂, c1, c2, tₙ#, rotC, ShiftRC, ShiftLC
end
