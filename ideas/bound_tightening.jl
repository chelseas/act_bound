# This will contain necessary functions for activation-(partition)-based bound tightening
using JuMP
using Gurobi
using NeuralVerification # Just using for a lightweight neural network representation
using LazySets
using Combinatorics
using LinearAlgebra
using IntervalArithmetic
using Plots

function take_max(bounds::Hyperrectangle)
    """ given a set of pre-activation bounds, return post-activation bounds by taking max with 0
    """
    new_low = max.(low(bounds), 0.0)
    new_high = max.(high(bounds), 0.0)
    return Hyperrectangle(low=new_low, high=new_high)
end

function tighten_bounds(net::NeuralVerification.Network, domain::Hyperrectangle)
    """ given a set of inputs to a layer, tighten the bounds on each neuron in the layer
    by considering activation patterns of other neurons in the layer
    """
    # b = init_bounds(net, domain)
    bounds = NeuralVerification.get_bounds(Ai2z(), net, domain; before_act = true)
    infos = []
    # intra-layer tightening
    for (i, layer) in enumerate(net.layers) 
        println("Tightening bounds for layer $i")
        input_bounds = bounds[i]
        if i > 1
            # bookmark: take max
            input_bounds = take_max(input_bounds) # turn pre-act into post act bounds
        end
        output_bounds = bounds[i+1]
        info = intra_layer_tighten(layer, input_bounds, output_bounds, i)
        push!(infos, info)
    end
    # inter-layer tightening? 
    ub_gaps = [infos[i][1] for i in 1:length(net.layers)]
    lb_gaps = [infos[i][2] for i in 1:length(net.layers)]
    stabs = [infos[i][3] for i in 1:length(net.layers)]
    ub_gap_fracs = [infos[i][4] for i in 1:length(net.layers)]
    lb_gap_fracs = [infos[i][5] for i in 1:length(net.layers)]
    return (ub_gaps, lb_gaps, stabs, ub_gap_fracs, lb_gap_fracs)
end

function unstable_neurons(bounds::Vector{<:Hyperrectangle})
    """ return indices of unstable neurons in a layer given bounds
    """
    unstable = []
    for b in bounds
        push!(unstable, unstable_neurons(b))
    end
    return unstable
end

function unstable_neurons(bounds::Hyperrectangle)
    return low(bounds) .< 0 .&& high(bounds) .> 0
end

function intra_layer_tighten(layer::NeuralVerification.Layer, input_bounds, output_bounds, i)
    """ tighten bounds on layer `i` considering activation patterns of other neurons in the same layer
    """
    n_nodes_i, n_nodes_im1 = size(layer.weights)
    unstable_out = unstable_neurons(output_bounds) 
    # only A has to be unstable to affect B
    perms = [p for p in permutations(1:n_nodes_i, 2) if unstable_out[p[1]]] # filter perms 
    println("perms: $perms")
    ub_gaps, lb_gaps, stab, ub_gap_fracs, lb_gap_fracs = [], [], 0, [], []
    for perm in perms
        println("considering permutation $perm")
        ub_gap_i, lb_gap_i, stab_i, ub_gf_i, lb_gf_i = solve_intra_layer_2(layer, input_bounds, output_bounds, i, perm)
        push!(ub_gaps, ub_gap_i)
        push!(lb_gaps, lb_gap_i)
        stab += stab_i
        push!(ub_gap_fracs, ub_gf_i)
        push!(lb_gap_fracs, lb_gf_i)
    end
    return (ub_gaps, lb_gaps, stab, ub_gap_fracs, lb_gap_fracs)
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

function get_interval_bounds(W_row::Vector{Float64}, bias_i::Float64, input_bounds::Hyperrectangle)
    """ get interval bounds for a neuron given weights, bias, and input bounds
    """
    ints = [interval(low(input_bounds)[j], high(input_bounds)[j]) for j in 1:length(W_row)]
    return sum(W_row .* ints) + bias_i
end

function solve_intra_layer_2(layer::NeuralVerification.Layer, input_bounds, output_bounds, i, combo)
    """ solve for bounds on layer `i` considering activation patterns of 2 neurons in `combo`
    """
    # Consider 2 nodes: node A and node B 
    # nodeA is combo[1], nodeB is combo[2]
    # consider how A affects B 
    W = layer.weights
    bias = layer.bias
    n_nodes_i, n_nodes_im1 = size(W)
    # first bounds in array are actually domain (bounds_0)
    Aind = combo[1]
    Bind = combo[2]
    # A -> B (how A affects B)
    function run_opt(on::Bool)
        m = Model(Gurobi.Optimizer)
        set_silent(m)
        # create variables for the nodes at layer i-1
        xvars = @variable(m, x[1:n_nodes_im1])
        add_var_bounds(m, xvars, input_bounds)
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
    println("Orginal bounds for neuron A: [", low(output_bounds)[Aind], ",", high(output_bounds)[Aind])
    println("Orginal bounds for neuron B: [", low(output_bounds)[Bind], ",", high(output_bounds)[Bind])
    println("Bounds for neuron B when A is off: [$LB_of_B_when_A_off, $UB_of_B_when_A_off]")
    println("Bounds for neuron B when A is on: [$LB_of_B_when_A_on, $UB_of_B_when_A_on]")
    blow = low(output_bounds)[Bind]
    bhi = high(output_bounds)[Bind]
    bint = get_interval_bounds(W[Bind, :], bias[Bind], input_bounds)
    bmin = min(blow, bint.lo)
    bmax = max(bhi, bint.hi)
    println("B bounds from int arithmetic are [$bmin, $bmax]")
    @assert(bmin <= LB_of_B_when_A_off || blow ≈ LB_of_B_when_A_off)
    @assert(bmin <= LB_of_B_when_A_on || blow ≈ LB_of_B_when_A_on)
    @assert(bmax >= UB_of_B_when_A_off || bhi ≈ UB_of_B_when_A_off)
    @assert(bmax >= UB_of_B_when_A_on || bhi ≈ UB_of_B_when_A_on)
    # calculate how much bound is tightened by 
    # we want to greatest LB and the smallest UB to calculate the gaps: (u - t^*_{UB}) and (t^*_{LB} - l)
    ub_gap = bhi - min(UB_of_B_when_A_off, UB_of_B_when_A_on)
    ub_gap = max(ub_gap, 0.0) # incase bounds we started with were tighter than naive LP solution
    ub_gap_frac = ub_gap / (bhi - blow)
    lb_gap = max(LB_of_B_when_A_off, LB_of_B_when_A_on) - blow
    lb_gap = max(lb_gap, 0.0)
    lb_gap_frac = lb_gap / (bhi - blow)
    # @assert(ub_gap >= 0)
    # @assert(lb_gap >= 0)
    println("UB gap: $ub_gap")
    println("LB gap: $lb_gap")
    # tally how many relus are stabilized ("stabilizing constraint" is added)
    stab = Int(LB_of_B_when_A_off*UB_of_B_when_A_off >= 0 && blow*bhi < 0 ) + Int(LB_of_B_when_A_on*UB_of_B_when_A_on >= 0 && blow*bhi < 0) # check if have same sign
    println("Number of stabilizing constraints by considering neuron A: $stab")

    return ub_gap, lb_gap, stab, ub_gap_frac, lb_gap_frac
end

x_lead = [90,110]
v_lead = [32, 32.2]
gamma_lead = [0,0]
x_ego = [10,11]
v_ego = [30, 30.2]
gamma_ego = [0, 0]
var_list = [x_lead, v_lead, gamma_lead, x_ego, v_ego, gamma_ego]
domain = Hyperrectangle(
    low=[v_range[1] for v_range in var_list], 
    high=[v_range[2] for v_range in var_list]
    )
nnet_name = "acc_controller.nnet"

# domain = Hyperrectangle(low=[-4.0, 2.0], high=[6.1, 7.1])
# nnet_name = "avoid_robot_network_clamped.nnet"
# nnet_name = "nn-nav-set.nnet"
# domain = Hyperrectangle(low=[-3.5, -3.5, -1., -2.], high=[3.5, 3.5, 3., 1.])
net = read_nnet(nnet_name)
(ub_gaps, lb_gaps, stabs, ub_gap_fracs, lb_gap_fracs) = tighten_bounds(net, domain)

# plot stuff 
pl1 = histogram(ub_gaps[1], bins=20, title="UB Gaps Layer 1", xlabel="UB Gap", ylabel="Count")
pl2 = histogram(ub_gaps[2], bins=20, title="UB Gaps Layer 2", xlabel="UB Gap", ylabel="Count")
pl3 = histogram(lb_gaps[1], bins=20, title="LB Gaps Layer 1", xlabel="LB Gap", ylabel="Count")
pl4 = histogram(lb_gaps[2], bins=20, title="LB Gaps Layer 2", xlabel="LB Gap", ylabel="Count")
pl5 = plot(pl1, pl3, pl2, pl4, layout=(2,2));
savefig(pl5, nnet_name*"bound_gaps_histograms.png")

println("Total number of stabilizing constraints added: ", stabs)

pl1 = histogram(ub_gap_fracs[1], bins=20, title="UB Gap Fracs Layer 1", xlabel="UB Gap Fraction", ylabel="Count")
pl2 = histogram(ub_gap_fracs[2], bins=20, title="UB Gap Fracs Layer 2", xlabel="UB Gap Frac", ylabel="Count")
pl3 = histogram(lb_gap_fracs[1], bins=20, title="LB Gap Fracs Layer 1", xlabel="LB Gap Fraction", ylabel="Count")
pl4 = histogram(lb_gap_fracs[2], bins=20, title="LB Gap Fracs Layer 2", xlabel="LB Gap Frac", ylabel="Count")
pl5 = plot(pl1, pl3, pl2, pl4, layout=(2,2));
display(pl5)
savefig(pl5, nnet_name*"bound_gap_fracs_histograms.png")



