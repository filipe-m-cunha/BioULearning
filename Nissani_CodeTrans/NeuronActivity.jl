function NeuronActivity(x, w, θₜ)
    wx = dot(w, x)
    y = ((sign(wx - θₜ) + 1)/2).*(wx - θₜ)
    return wx, y
end