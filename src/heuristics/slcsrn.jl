## Neural heuristic using CSRN domain-independent architecture 
export slCSRNHeuristic 

mutable struct slCSRNHeuristic <: Heuristic
    init_sl_layers::Array{Float32,3} # Initial state in the sl-layered form 
    problem::Problem # Problem definition for further computation 
    coord_dict::Dict # Mapping between objects representing positions in a grid and coordinates 
    object_info::Tuple # Information about objects used to update state, compute cache keys, etc. (depends on the domain)
    
    cell # slCSRN cell 
    no_iters::Int64 # Number of recurrent iterations 
    multislice::Bool # if the network is multislice or not 
    slice_encoding::Array{Float32,1} # slice encoding -> necessary for the CSRN 

    run_cell_faster::Function # Function to run the slCSRN 
    get_cache_key::Function # Function to return key for the heuristic cache 
    change_sl_for_state::Function # Function that updates sl state representation based on current state 

    slCSRN_cache::Dict # Cache for computation of the heuristic

    graph::SymbolicPlanners.PlanningGraph # Precomputed planning graph - not sure I need it 
    pre_key::Tuple{UInt64,UInt64} # Precomputation hash
    goal_idxs::Set{Int} # Precomputed list of goal indices - not sure I need it 
    
    slCSRNHeuristic() = new()
end

Base.hash(heuristic::slCSRNHeuristic, h::UInt) = hash(slCSRNHeuristic, h)

function precompute!(h::slCSRNHeuristic, domain::Domain, state::State, spec::Specification)
    # Check if cache has already been computed
    if is_precomputed(h, domain, state, spec) return h end
    # Precomputed data is unique to each domain and specification
    h.pre_key = (objectid(domain), objectid(spec))
    # Build planning graph and find goal condition indices
    goal_conds = PDDL.to_cnf_clauses(get_goal_terms(spec))
    h.graph = build_planning_graph(domain, state, goal_conds)
    h.goal_idxs = Set(findall(c -> c in goal_conds, h.graph.conditions))
    return h
end

function is_precomputed(h::slCSRNHeuristic, domain::Domain, state::State, spec::Specification)
    return (isdefined(h, :graph) &&
            objectid(domain) == h.pre_key[1] &&
            objectid(spec) == h.pre_key[2])
end

# Initialize the slCSRN heuristic structure with neural network necessities - onelayer
function init_slCSRN_heur(h::slCSRNHeuristic, init_sl_layers::Array{Float32,3}, cell, no_iters::Int64, problem::Problem, coord_dict::Dict, 
    object_info::Tuple, get_cache_key::Function, change_sl_for_state::Function, run_cell_faster::Function)
    h.init_sl_layers = init_sl_layers
    h.cell = cell
    h.no_iters = no_iters
    h.slCSRN_cache = Dict()
    h.problem = problem
    h.coord_dict = coord_dict
    h.object_info = object_info
    h.get_cache_key = get_cache_key
    h.change_sl_for_state = change_sl_for_state
    h.run_cell_faster = run_cell_faster
    h.multislice = false 

    return h
end

# Initialize the slCSRN heuristic structure with neural network necessities - multi-layer 
function init_slCSRN_heur(h::slCSRNHeuristic, init_sl_layers::Array{Float32,3}, cell, no_iters::Int64, problem::Problem, coord_dict::Dict, 
    object_info::Tuple, get_cache_key::Function, change_sl_for_state::Function, run_cell_faster::Function, slice_encoding::Array{Float32,1})
    h.init_sl_layers = init_sl_layers
    h.cell = cell
    h.no_iters = no_iters
    h.slCSRN_cache = Dict()
    h.problem = problem
    h.coord_dict = coord_dict
    h.object_info = object_info
    h.get_cache_key = get_cache_key
    h.change_sl_for_state = change_sl_for_state
    h.run_cell_faster = run_cell_faster
    h.multislice = true
    h.slice_encoding = slice_encoding 

    return h
end

# Utility function to debinerize newly created layers and keeping the static ones 
function debinerize_layers(input::Array{Float32,3})
    one_ixs = CartesianIndices(input)[input .== 1.]
    zero_ixs = CartesianIndices(input)[input .== 0.]

    input[one_ixs] .= 0.9
    input[zero_ixs] .= 0.1 

    return input 
end

function compute(h::slCSRNHeuristic, domain::Domain, state::State, spec::Specification)
    # Precompute if necessary
    if !is_precomputed(h, domain, state, spec)
        precompute!(h, domain, state, spec)
    end

    # Change the input layers according to the current state 
    state_sl_layers = debinerize_layers(h.change_sl_for_state(h, state, h.init_sl_layers, h.coord_dict))
    k = h.get_cache_key(state_sl_layers)

    val = Float32.(0.9)
    ag = CartesianIndices(state_sl_layers[:,:,3])[state_sl_layers[:,:,3] .== val][1]

    # Check the cache based on the agent+box positions 
    if k in keys(h.slCSRN_cache)
        heur_layer = h.slCSRN_cache[k]
        return heur_layer[ag][1]
    else 
        # TODO: add function that returns a value from the slCSRN output -> different for different domains 
        if h.multislice
            output = sum(h.run_cell_faster(h.cell, state_sl_layers, h.no_iters, h.slice_encoding) .* state_sl_layers, dims=3)[:,:,1]
        else 
            output = sum(h.run_cell_faster(h.cell, state_sl_layers, h.no_iters) .* state_sl_layers, dims=3)[:,:,1]
        end
        h.slCSRN_cache[k] = output
        return output[ag][1]
    end    
end