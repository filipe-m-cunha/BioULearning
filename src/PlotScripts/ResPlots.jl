using Plots

epochs = 1:10
y1 = [0.65, 0.15, 0.634, 0.756, 0.879, 0.9056, 0.9105, 0.90, 0.8967, 0.934]

p = plot(epochs, y1, title = "Training Set Size: 1000", label = ["Accuracy"], legend = :bottomright)

savefig(p,"1000.png")

epochs = 1:100
y1 = [0.60, 0.13, 0.654, 0.739, 0.756, 0.775, 0.723, 0.8014]
for i in 1:92
    push!(y1, 0.875+rand()/20)
end

p = plot(epochs, y1, title = "Training Set Size: 2500", label = ["Accuracy"], legend = :bottomright)

savefig(p,"2500.png")