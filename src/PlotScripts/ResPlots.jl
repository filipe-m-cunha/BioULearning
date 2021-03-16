using Plots

epochs = 1:10
y1 = [0.89013, 0.8975, 0.8975, 0.8975, 0.8975, 0.8975, 0.8975, 0.8975, 0.8975, 0.8975]

p = plot(epochs, y1, title = "Training Set Size: 1000", label = ["Accuracy"], legend = :bottomright)

savefig(p,"1000.png")

epochs = 1:100
y1 = [0.89013]
for i in 1:99
    push!(y1, 0.8975)
end

p = plot(epochs, y1, title = "Training Set Size: 2500", label = ["Accuracy"], legend = :bottomright)

savefig(p,"2500.png")