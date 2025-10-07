using Gurobi
using JuMP
using LazySets
using LinearAlgebra
using IntervalArithmetic
using Plots

##########################
# x1 ∈ [-1,1], x2 ∈ [-1,1]
# consider the layer
# [ 1   -1][x1]    [-1]
# [-1    1][x2] +  [ 1]
##########################
W = [1. -1.; -1/4. 1/2.]
b = [-1/4.; 0.]
x1 = interval(-1, 1)
x2 = interval(-1, 1)

xhat1 = W[1,1]*x1 + W[1,2]*x2 + b[1]  # pre-activation of neuron 1
xhat2 = W[2,1]*x1 + W[2,2]*x2 + b[2]  # pre-activation of neuron 2

m = Model(Gurobi.Optimizer)
x1v = @variable(m, -1 <= x1v <= 1)
x2v = @variable(m, -1 <= x2v <= 1)
@objective(m, Min, W[2,1]*x1v + W[2,2]*x2v + b[2])
@constraint(m, W[1,1]*x1v + W[1,2]*x2v + b[1] >= 0)  # neuron 1 active
optimize!(m)
LB = objective_value(m)
println("LB for neuron 2 when neuron 1 active: $LB")
@objective(m, Max, W[2,1]*x1v + W[2,2]*x2v + b[2])
optimize!(m)
UB = objective_value(m)
println("UB for neuron 2 when neuron 1 active: $UB")
#### solving LP right now but I think can be solved in closed form ?
#~~~~~~~~~~~~~~~~~~~~~~~
# plot hyperplanes 
h1 = Hyperplane(W[1,:], -b[1])  # hyperplane for neuron 1
h1h = HalfSpace(W[1,:], -b[1])  # halfspace for neuron 1 active
h2 = Hyperplane(W[2,:], -b[2])  # hyperplane
h2h = HalfSpace(W[2,:], -b[2])  # halfspace for neuron 2 active
p = plot(h1)
    plot!(h1h, label="neuron 1 active")
    plot!(h2)
    plot!(h2h, label="neuron 2 active")
# ~~~~~~~~~~~~~~~~~~~~~~~
# plot relu space of neuron 2
pr = plot([xhat2.lo, 0.0],[0.0, 0.0], label="neuron 2 inactive", xlims=(-1,1), ylims=(-1,1))
plot!([0.0, xhat2.hi],[0.0, xhat2.hi], label="neuron 2 active") 
# y ≦ z - LB(1-δ)
xrange = -1:0.1:1
plot!([xrange...], [(xrange .- xhat2.lo)...], label="y ≤ z - LB(1-δ)", color=:orange, fillrange=-100, alpha=0.1)
#
# y ≦ UB δ
plot!([xrange...], [xhat2.hi.*ones(length(xrange))...], label="y ≤ UB δ", color=:green, fillrange=-100, alpha=0.05)
hline!([0.125], linestyle=:dash, label="UB when neuron 1 active")

# LB when δ₁ inactive = -0.3125
plot!(xrange, xrange .- (-0.3125), label="y ≤ z - LB(1-δ) when neuron 1 inactive", color=:red, linestyle=:dash)

### Ok, so bounds change based upon activation status of other neurons in same layer. 
# can we formulate this as a constraint w/ delta variable and check solve speed? 
# 
