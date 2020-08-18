# Test that planners correctly solve simple problems

@testset "Planners" begin

# Load domains and problems
path = joinpath(dirname(pathof(SymbolicPlanners)), "..", "domains", "gridworld")
gridworld = load_domain(joinpath(path, "domain.pddl"))
gw_problem = load_problem(joinpath(path, "problem-1.pddl"))
gw_state = init_state(gw_problem)

path = joinpath(dirname(pathof(SymbolicPlanners)), "..", "domains", "doors-keys-gems")
doors_keys_gems = load_domain(joinpath(path, "domain.pddl"))
dkg_problem = load_problem(joinpath(path, "problem-1.pddl"))
dkg_state = init_state(dkg_problem)

path = joinpath(dirname(pathof(SymbolicPlanners)), "..", "domains", "blocksworld")
blocksworld = load_domain(joinpath(path, "domain.pddl"))
bw_problem = load_problem(joinpath(path, "problem-0.pddl"))
bw_state = init_state(bw_problem)

@testset "Breadth-First Planner" begin

planner = BreadthFirstPlanner()
plan, traj = planner(gridworld, gw_state, gw_problem.goal)
@test satisfy(gw_problem.goal, traj[end], gridworld)[1] == true
@test plan == @pddl("down", "down", "right", "right", "up", "up")

plan, traj = planner(doors_keys_gems, dkg_state, dkg_problem.goal)
@test satisfy(dkg_problem.goal, traj[end], doors_keys_gems)[1] == true
@test plan == @pddl("(down)", "(pickup key1)", "(down)", "(unlock key1 right)",
                    "(right)", "(right)", "(up)", "(up)", "(pickup gem1)")

plan, traj = planner(blocksworld, bw_state, bw_problem.goal)
@test satisfy(bw_problem.goal, traj[end], blocksworld)[1] == true
@test plan == @pddl("(pick-up a)", "(stack a b)", "(pick-up c)", "(stack c a)")

end

@testset "Uniform Cost Planner" begin

clear_heuristic_cache!()
clear_available_action_cache!()

planner = UniformCostPlanner()
plan, traj = planner(gridworld, gw_state, gw_problem.goal)
@test satisfy(gw_problem.goal, traj[end], gridworld)[1] == true
@test plan == @pddl("down", "down", "right", "right", "up", "up")

planner = UniformCostPlanner()
plan, traj = planner(doors_keys_gems, dkg_state, dkg_problem.goal)
@test satisfy(dkg_problem.goal, traj[end], doors_keys_gems)[1] == true
@test plan == @pddl("(down)", "(pickup key1)", "(down)", "(unlock key1 right)",
                    "(right)", "(right)", "(up)", "(up)", "(pickup gem1)")

planner = UniformCostPlanner()
plan, traj = planner(blocksworld, bw_state, bw_problem.goal)
@test satisfy(bw_problem.goal, traj[end], blocksworld)[1] == true
@test plan == @pddl("(pick-up a)", "(stack a b)", "(pick-up c)", "(stack c a)")

end

@testset "Greedy Planner" begin

clear_heuristic_cache!()
clear_available_action_cache!()

planner = GreedyPlanner(ManhattanHeuristic(@pddl("xpos", "ypos")))
plan, traj = planner(gridworld, gw_state, gw_problem.goal)
@test satisfy(gw_problem.goal, traj[end], gridworld)[1] == true
@test plan == @pddl("down", "down", "right", "right", "up", "up")

planner = GreedyPlanner(GoalCountHeuristic())
plan, traj = planner(doors_keys_gems, dkg_state, dkg_problem.goal)
@test satisfy(dkg_problem.goal, traj[end], doors_keys_gems)[1] == true
@test plan == @pddl("(down)", "(pickup key1)", "(down)", "(unlock key1 right)",
                    "(right)", "(right)", "(up)", "(up)", "(pickup gem1)")

planner = GreedyPlanner(HAdd())
plan, traj = planner(blocksworld, bw_state, bw_problem.goal)
@test satisfy(bw_problem.goal, traj[end], blocksworld)[1] == true
@test plan == @pddl("(pick-up a)", "(stack a b)", "(pick-up c)", "(stack c a)")

end

@testset "A* Planner" begin

clear_heuristic_cache!()
clear_available_action_cache!()

planner = AStarPlanner(ManhattanHeuristic(@pddl("xpos", "ypos")))
plan, traj = planner(gridworld, gw_state, gw_problem.goal)
@test satisfy(gw_problem.goal, traj[end], gridworld)[1] == true
@test plan == @pddl("down", "down", "right", "right", "up", "up")

planner = AStarPlanner(GoalCountHeuristic())
plan, traj = planner(doors_keys_gems, dkg_state, dkg_problem.goal)
@test satisfy(dkg_problem.goal, traj[end], doors_keys_gems)[1] == true
@test plan == @pddl("(down)", "(pickup key1)", "(down)", "(unlock key1 right)",
                    "(right)", "(right)", "(up)", "(up)", "(pickup gem1)")

planner = AStarPlanner(HAdd())
plan, traj = planner(blocksworld, bw_state, bw_problem.goal)
@test satisfy(bw_problem.goal, traj[end], blocksworld)[1] == true
@test plan == @pddl("(pick-up a)", "(stack a b)", "(pick-up c)", "(stack c a)")

end

@testset "Backward Planner" begin

clear_heuristic_cache!()
clear_relevant_action_cache!()

planner = BackwardPlanner(heuristic=HAddR())
plan, traj = planner(blocksworld, bw_state, bw_problem.goal)
@test issubset(traj[1], bw_state) == true
@test plan == @pddl("(pick-up a)", "(stack a b)", "(pick-up c)", "(stack c a)")

end

end
