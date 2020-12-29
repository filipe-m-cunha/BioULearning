function NeuronActivity(x::Array{Float64, 1}, w::Array{Float64, 1}, θₜ::Array{Float64, 1})
    wx = dot(w, x)
    y = ((sign(wx - θₜ) + 1)/2).*(wx - θₜ)
    return wx, y
end