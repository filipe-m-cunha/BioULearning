function determine_ineq(A::Array{Float64, 1} , B::Float64)
    s = []
    for i in 1:length(A)
        if A[i] == B
            append!(s, 1)
        else
            append!(s, 0)
        end
    end
    s
end