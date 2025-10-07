# This will contain necessary functions for activation-(partition)-based bound tightening
using JuMP
using Gurobi
using NeuralVerification # Just using for a lightweight neural network representation
using LazySets
using Combinatorics
using LinearAlgebra

domain = Hyperrectangle(low=[-4.0, 2.0], high=[6.1, 7.1])
net = read_nnet("avoid_robot_network_clamped.nnet")
function tighten_bounds(net::NeuralVerification.Network, domain::Hyperrectangle)
    """ given a set of inputs to a layer, tighten the bounds on each neuron in the layer
    by considering activation patterns of other neurons in the layer
    """
    # b = init_bounds(net, domain)
    b = NeuralVerification.get_bounds(Ai2z(), net, domain; before_act = true)
    b2 = []
    # intra-layer tightening
    for (i, layer) in enumerate(net.layers) 
        b2_i = intra_layer_tighten(layer, domain, b, i)
        push!(b2, b2_i)
    end
    # inter-layer tightening? 
    return b, b2
end

function intra_layer_tighten(layer::NeuralVerification.Layer, domain, b, i)
    """ tighten bounds on layer `i` considering activation patterns of other neurons in the same layer
    """
    n_nodes_i, n_nodes_im1 = size(layer.weights)
    combos = [c for c in combinations(1:n_nodes_i, 2)] # n choose 2
    new_bounds = []
    for combo in combos
        println("considering combo $combo")
        new_bounds_i = solve_intra_layer_2(layer, b, i, combo, domain)
        push!(new_bounds, new_bounds_i)
        println("new bounds: $new_bounds")
    end
    return new_bounds
end

function add_var_bounds(vars::Vector{VariableRef}, bounds::Hyperrectangle)
     for j in 1:n_nodes_im1
        @constraint(m, low(b)[j] <= xvars[j] <= high(b)[j])
    end
end
function add_var_bounds(vars::Vector{VariableRef}, bounds::Vector{<:Vector})
    for j in 1:n_nodes_im1
        @constraint(m, b[j][1] <= xvars[j] <= b[j][2])
    end
end

function solve_intra_layer_2(layer::NeuralVerification.Layer, b, i, combo, domain)
    """ solve for bounds on layer `i` considering activation patterns of 2 neurons in `combo`
    """
    # Consider 2 nodes: node A and node B 
    # nodeA is combo[1], nodeB is combo[2]
    # consider how A affects B 
    W = layer.weights
    bias = layer.bias
    n_nodes_i, n_nodes_im1 = size(W)
    if i == 1
        bounds_i = b[i]
        bounds_im1 = domain 
    else
        bounds_i = b[i]
        bounds_im1 = b[i-1]
    end
    Aind = combo[1]
    Bind = combo[2]
    # A -> B (how A affects B)
    m = Model(Gurobi.Optimizer)
    set_silent(m)
    # create variables for the nodes at layer i-1
    xvars = @variable(m, x[1:n_nodes_im1])
    add_var_bounds(xvars, bounds_im1)
    nodeAexpr = dot(W[Aind, :], xvars) + bias[Aind]
    nodeBexpr = dot(W[Bind, :], xvars) + bias[Bind]
    ###################################################
    # now solve for UB and LB of B when A is off 
    ###################################################
    # first UB of B when A is off 
    Aoff = @constraint(m, nodeAexpr <= 0)
    @objective(m, Max, nodeBexpr)
    optimize!(m)
    UB_of_B_when_A_off = objective_value(m)
    # then solve for LB of  B when A off 
    @objective(m, Min, nodeBexpr)
    optimize!(m)
    LB_of_B_when_A_off = objective_value(m)
    ##################################################
    # now solve for UB and LB of B when A is on  
    ###################################################
    delete(m, Aoff)
    Aon = @constraint(m, nodeAexpr >= 0)
    # first UB of B when A is on
    @objective(m, Max, nodeBexpr)
    optimize!(m)
    UB_of_B_when_A_on = objective_value(m)
    # then solve for LB of  B when A on
    @objective(m, Min, nodeBexpr)
    optimize!(m)
    LB_of_B_when_A_off = objective_value(m)
    ####################################################
    println("Orginal bounds for neuron B: ", b[i][Bind])
    println("Bounds for neuron B when A is off: [$LB_of_B_when_A_off, $UB_of_B_when_A_off]")
    println("Bounds for neuron B when A is on: [$LB_of_B_when_A_on, $UB_of_B_when_A_on]")
    return nothing 
end








    
    
    # x1 = b[i-1][1]
    # x2 = b[i-1][2]
    # ...
    #zhat1 = W[1,1]*x1 + W[1,2]*x2 + bias[1]  # pre-activation of neuron 1
    #zhat2 = W[2,1]*x1 + W[2,2]*x2 + bias[2]  # pre-activation of neuron 2
    # ...

    m = Model(Gurobi.Optimizer)
    set_silent(m)
    xvars = []
    for j in 1:n_nodes_im1
        @variable(m, b[i-1][j][1] <= xvars[j= j] <= b[i-1][j][2])
    end
    # set activation pattern for neurons in combo
    for j in combo
        @constraint(m, sum(W[j,k]*xvars[k] for k in 1:n_nodes_im1) + bias[j] >= 0)  # neuron j active
    end
    new_bounds = []
    for j in 1:n_nodes_i
        @objective(m, Min, sum(W[j,k]*xvars[k] for k in 1:n_nodes_im1) + bias[j])
        optimize!(m)
        LB = objective_value(m)
        @objective(m, Max, sum(W[j,k]*xvars[k] for k in 1:n_nodes_im1) + bias[j])
        optimize!(m)
        UB = objective_value(m)
        push!(new_bounds, (LB, UB))
        println("neuron $j: ($LB, $UB)")
    end
    return new_bounds
end
