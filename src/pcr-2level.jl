using StatsBase
using CairoMakie
using LinearAlgebra
using Distributions
using DataFrames
using DifferentialEquations
set_theme!(theme_minimal())

get_index(n_R, n_C) = (1:n_R, n_R+1:n_R+n_C)
function get_index(p)
    n_C = length(p[3])
    n_R = length(p[4])
    get_index(n_R, n_C)
end

richness(p) = sum(length.(get_index(p)))

function pcr!(du, u, p, _)
    c, e, m_C, r, K, m_R, theta = p
    r_ind, c_ind = get_index(p)
    R, C = u[r_ind], u[c_ind]
    # Resource dynamics.
    for (i, res) in enumerate(r_ind)
        du[res] =
            (
                r[i] * (1 - R[i] / K[i]) -
                sum(c[i, :] .* C ./ (1 .+ theta * C .+ 0.1R[i])) - m_R[i]
            ) * R[i]
    end
    # Consumers dynamics.
    for (i, cons) in enumerate(c_ind)
        du[cons] =
            (sum(e[:, i] .* c[:, i] .* R / (1 .+ theta * C[i] .+ 0.1R)) - m_C[i]) * C[i]
    end
end

function get_R_alone(p)
    r_ind, _ = get_index(p)
    R_alone = zeros(length(r_ind))
    for (i, res) in enumerate(r_ind)
        u0 = zeros(richness(p))
        u0[res] = 1
        R_alone[i] = get_sol(p; u0)[res]
    end
    R_alone
end

function get_C_alone(p)
    r_ind, c_ind = get_index(p)
    C_alone = zeros(length(c_ind))
    for (i, cons) in enumerate(c_ind)
        u0 = zeros(richness(p))
        u0[r_ind] .= 1
        u0[cons] = 1
        C_alone[i] = get_sol(p; u0)[cons]
    end
    C_alone
end

function get_sol(p; tspan = (0, 500), u0 = fill(1, richness(p)))
    prob = ODEProblem(pcr!, u0, tspan, p)
    sol = DifferentialEquations.solve(prob)
    sol[end]
end

function generate_community(
    n,
    r_i,
    K_i,
    mr_i,
    c_off,
    c_diag,
    e_i,
    m_i,
    theta,
    threshold = 0.1,
    maxiter = 100,
)
    u_eq = zeros(2n)
    iter = 0
    while any(u_eq .< threshold) && iter < maxiter
        r = rand(r_i, n)
        K = rand(K_i, n)
        c = rand(c_off, n, n)
        c[diagind(c)] .= c_diag
        e = rand(e_i, n, n)
        m = rand(m_i, n)
        mr = rand(mr_i, n)
        global p = (c, e, m, r, K, mr, theta)
        u_eq = get_sol(p)
        iter += 1
    end
    @info iter
    p
end

n = 5
r_i = Normal(10, 0)
K_i = Normal(10, 0)
mr_i = Normal(0, 0)
c_off = Uniform(0.1, 0.8)
c_diag = 1
e_i = Normal(0.1, 0.0)
m_i = Normal(0.1, 0.0)
theta_list = repeat(collect(0.12:0.01:0.3), 1)
p_list = [
    generate_community(n, r_i, K_i, mr_i, c_off, c_diag, e_i, m_i, theta) for
    theta in theta_list
]

function initiate_df()
    df = DataFrame(;
        c_id = Int64[],
        sp_id = Int64[],
        sp_type = Symbol[],
        sl = Float64[],
        s = Float64[],
        tl_perturbed = Symbol[],
        theta = Float64[],
    )
end

scenarios = [:community, :targeted]
n_rep = 10
D = LogNormal(log(0.05), 0.7)
df_list = Any[nothing for _ in eachindex(p_list)]
for i in eachindex(p_list)
    df_thread = initiate_df()
    p = p_list[i]
    r_ind, c_ind = get_index(p)
    n_R, n_C = length.(get_index(p))
    u_eq = get_sol(p)
    R_alone = get_R_alone(p)
    C_alone = get_C_alone(p)
    SL = vcat(u_eq[r_ind] ./ R_alone, u_eq[c_ind] ./ C_alone)
    c, e, m, r, K, mr, theta = deepcopy(p)
    for k in 1:n_rep, tl_perturbed in scenarios
        if tl_perturbed == :community
            global p_press = (c, e, m .+ rand(D, n), r, K, mr .+ rand(D, n), theta)
            R_alone_press = get_R_alone(p_press)
            C_alone_press = get_C_alone(p_press)
            diff_R_alone = (R_alone_press .- R_alone) ./ R_alone
            diff_C_alone = (C_alone_press .- C_alone) ./ C_alone
            diff_u_alone = vcat(diff_R_alone, diff_C_alone)
            u_press = get_sol(p_press)
            diff_u = (u_press .- u_eq) ./ u_eq
            s_press = diff_u ./ diff_u_alone
        elseif tl_perturbed == :targeted
            s_press = zeros(richness(p))
            for sp in eachindex(s_press)
                m_cat = vcat(mr, m)
                m_cat[sp] += rand(D)
                mr_press, m_press = m_cat[r_ind], m_cat[c_ind]
                global p_press = (c, e, m_press, r, K, mr_press, theta)
                if sp in r_ind
                    idx = findfirst(==(sp), r_ind)
                    R_alone_press = get_R_alone(p_press)[idx]
                    diff_B_alone = (R_alone_press - R_alone[idx]) / R_alone[idx]
                elseif sp in c_ind
                    idx = findfirst(==(sp), c_ind)
                    C_alone_press = get_C_alone(p_press)[idx]
                    diff_B_alone = (C_alone_press - C_alone[idx]) / C_alone[idx]
                else
                    @error "Wrong species index."
                end
                u_press = get_sol(p_press)[sp]
                diff_u = (u_press - u_eq[sp]) / u_eq[sp]
                s_press[sp] = diff_u / diff_B_alone
            end
        else
            @error "Invalid scenario."
        end
        for (sp_id, s) in enumerate(s_press)
            sp_type = sp_id ∈ r_ind ? :r : :c
            push!(df_thread, (i, sp_id, sp_type, SL[sp_id], s, tl_perturbed, theta))
        end
    end
    df_list[i] = df_thread
end
df = vcat(df_list...)

df_mean = combine(groupby(df, [:c_id, :sl, :sp_type, :tl_perturbed, :theta]), :s => mean)
colors = Dict(:r => :black, :c => :gray51, :p => :gray76)
inch = 96
pt = 4 / 3
cm = inch / 2.54
width = 13cm
alpha = 1
fig = Figure(; size = (width, 0.7width), fontsize = 10pt);
axes = Array{Any}(undef, 2, 2)
markersize = 5
for (j, scenario) in enumerate(scenarios)
    df_scenario = subset(df_mean, :tl_perturbed => ByRow(==(scenario)))
    for (i, type) in enumerate([:c, :r])
        axes[i, j] = ax = Axis(fig[i, j]; xlabel = "SL", ylabel = "Stability to press")
        color = colors[type]
        df_type = subset(df_scenario, :sp_type => ByRow(==(type)))
        scatter!(df_type.sl, 1 .- df_type.s_mean; color = df_type.theta)
    end
end
Colorbar(fig[1:2, 3]; label = "Interference", limits = extrema(df_mean.theta))
axes[1, 1].xlabel = ""
axes[1, 2].xlabel = ""
axes[1, 2].ylabel = ""
axes[2, 2].ylabel = ""
fig

save("figures/two-levels-theta.svg", fig)

