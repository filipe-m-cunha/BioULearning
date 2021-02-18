function NeuronActivity(x::Array{Float64, 1}, w::Array{Float64, 1}, θ::Float64)
    wx = transpose(w)*x
    y = ((sign(wx - θ) + 1)/2)*(wx - θ)
    return wx, y
end