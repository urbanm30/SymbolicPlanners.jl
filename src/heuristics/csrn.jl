## Neural heuristic using CSRN domain-independent architecture 
export CSRNHeuristic 

mutable struct CSRNHeuristic <: Heuristic
    csrn_out
    f_to_ix::Dict # indexing 
    graph::SymbolicPlanners.PlanningGraph # Precomputed planning graph - not sure I need it 
    pre_key::Tuple{UInt64,UInt64} # Precomputation hash
    goal_idxs::Set{Int} # Precomputed list of goal indices - not sure I need it 
    CSRNHeuristic() = new()
end

Base.hash(heuristic::CSRNHeuristic, h::UInt) = hash(CSRNHeuristic, h)

function precompute!(h::CSRNHeuristic, domain::Domain, state::State, spec::Specification)
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

function is_precomputed(h::CSRNHeuristic, domain::Domain, state::State, spec::Specification)
    return (isdefined(h, :graph) &&
            objectid(domain) == h.pre_key[1] &&
            objectid(spec) == h.pre_key[2])
end

# Put NN into the structure 
function init_cell(h::CSRNHeuristic, csrn_out, f_to_ix::Dict)
    h.csrn_out = csrn_out
    h.f_to_ix = f_to_ix
    return h
end

function compute(h::CSRNHeuristic, domain::Domain, state::State, spec::Specification)
    # Precompute if necessary
    if !is_precomputed(h, domain, state, spec)
        precompute!(h, domain, state, spec)
    end

    # Create in_state vector based on the state 
    in_state = zeros(1,length(h.csrn_out))
    for f in state.facts 
        if f in keys(h.f_to_ix)
            ix = h.f_to_ix[f]
            in_state[ix] = 1.
        end
    end

    # Binary combination 
    heur_val = h.csrn_out * in_state'

    return heur_val[1]
end