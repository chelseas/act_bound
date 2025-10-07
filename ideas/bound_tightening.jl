# This will contain necessary functions for activation-(partition)-based bound tightening
using JuMP
using Gurobi
using NeuralVerification # Just using for a lightweight neural network representation
using LazySets
using Combinatorics
using LinearAlgebra

function tighten_bounds(net::NeuralVerification.Network, domain::Hyperrectangle)
    """ given a set of inputs to a layer, tighten the bounds on each neuron in the layer
    by considering activation patterns of other neurons in the layer
    """
    # b = init_bounds(net, domain)
    bounds = NeuralVerification.get_bounds(Ai2z(), net, domain; before_act = true)
    bounds_new = []
    # intra-layer tightening
    for (i, layer) in enumerate(net.layers) 
        bn_i = intra_layer_tighten(layer, bounds, i)
        push!(bounds_new, bn_i)
    end
    # inter-layer tightening? 
    return bounds, bounds_new
end

function unstable_neurons(bounds::Vector{<:Hyperrectangle})
    """ return indices of unstable neurons in a layer given bounds
    """
    unstable = []
    for b in bounds
        push!(unstable, low(b) .< 0 .&& high(b) .> 0)
    end
    return unstable
end

function intra_layer_tighten(layer::NeuralVerification.Layer, bounds, i)
    """ tighten bounds on layer `i` considering activation patterns of other neurons in the same layer
    """
    n_nodes_i, n_nodes_im1 = size(layer.weights)
    unstable = unstable_neurons(bounds) # computes for whole network right now
    # only consider permutations of unstable neurons 
    # A must be unstable, but not B
    perms = [p for p in permutations(1:n_nodes_i, 2) if unstable[i+1][p[1]] ] # filter perms 
    new_bounds = []
    for perm in perms
        println("considering permutation $perm")
        new_bounds_i = solve_intra_layer_2(layer, bounds, i, perm)
        push!(new_bounds, new_bounds_i)
        println("new bounds: $new_bounds")
    end
    return new_bounds
end

function add_var_bounds(m, vars::Vector{VariableRef}, bounds::Hyperrectangle)
     for j in 1:length(vars)
        @constraint(m, low(bounds)[j] <= vars[j] <= high(bounds)[j])
    end
end
function add_var_bounds(m, vars::Vector{VariableRef}, bounds::Vector{<:Vector})
    for j in 1:length(vars)
        @constraint(m, bounds[j][1] <= vars[j] <= bounds[j][2])
    end
end

function solve_intra_layer_2(layer::NeuralVerification.Layer, bounds, i, combo)
    """ solve for bounds on layer `i` considering activation patterns of 2 neurons in `combo`
    """
    # Consider 2 nodes: node A and node B 
    # nodeA is combo[1], nodeB is combo[2]
    # consider how A affects B 
    W = layer.weights
    bias = layer.bias
    n_nodes_i, n_nodes_im1 = size(W)
    # first bounds in array are actually domain (bounds_0)
    bounds_i = bounds[i+1]
    bounds_im1 = bounds[i]
    Aind = combo[1]
    Bind = combo[2]
    # A -> B (how A affects B)
    function run_opt(on::Bool)
        m = Model(Gurobi.Optimizer)
        set_silent(m)
        # create variables for the nodes at layer i-1
        xvars = @variable(m, x[1:n_nodes_im1])
        add_var_bounds(m, xvars, bounds_im1)
        nodeAexpr = dot(W[Aind, :], xvars) + bias[Aind]
        nodeBexpr = dot(W[Bind, :], xvars) + bias[Bind]
        if on
            @constraint(m, nodeAexpr >= 0)
        else
            @constraint(m, nodeAexpr <= 0)
        end
        @objective(m, Max, nodeBexpr)
        optimize!(m)
        UB = objective_value(m)
        @objective(m, Min, nodeBexpr)
        optimize!(m)
        LB = objective_value(m)
        return LB, UB
    end
    ###################################################
    # now solve for UB and LB of B when A is off 
    ###################################################
    # first UB and LB of B when A is off 
    LB_of_B_when_A_off, UB_of_B_when_A_off = run_opt(false)
    ##################################################
    # now solve for UB and LB of B when A is on  
    ###################################################
    LB_of_B_when_A_on, UB_of_B_when_A_on = run_opt(true)
    ####################################################
    println("Orginal bounds for neuron A: [", low(bounds_i)[Aind], ",", high(bounds_i)[Aind])
    println("Orginal bounds for neuron B: [", low(bounds_i)[Bind], ",", high(bounds_i)[Bind])
    println("Bounds for neuron B when A is off: [$LB_of_B_when_A_off, $UB_of_B_when_A_off]")
    println("Bounds for neuron B when A is on: [$LB_of_B_when_A_on, $UB_of_B_when_A_on]")
    return nothing 
end

domain = Hyperrectangle(low=[-4.0, 2.0], high=[6.1, 7.1])
net = read_nnet("avoid_robot_network_clamped.nnet")
tighten_bounds(net, domain)







