using StatsBase
using GLV
using CairoMakie
using LinearAlgebra
using Distributions
using DataFrames
using DifferentialEquations
set_theme!(theme_minimal())

get_index(n_R, n_C, n_P) = (1:n_R, n_R+1:n_R+n_C, n_C+n_R+1:n_C+n_R+n_P)
function get_index(p)
    n_C = length(p[3])
    n_R = length(p[4])
    n_P = length(p[end-1])
    get_index(n_R, n_C, n_P)
end

richness(p) = sum(length.(get_index(p)))

function pcr!(du, u, p, _)
    c, e, m_C, r, K, m_R, c_p, e_p, m_P, theta = p
    r_ind, c_ind, p_ind = get_index(p)
    R, C, P = u[r_ind], u[c_ind], u[p_ind]
    # Resource dynamics.
    for (i, res) in enumerate(r_ind)
        du[res] =
            (
                r[i] * (1 - R[i] / K[i]) - sum(c[i, :] .* C ./ (1 .+ 0.5C .+ 0.1R[i])) -
                m_R[i]
            ) * R[i]
    end
    # Consumers dynamics.
    for (i, cons) in enumerate(c_ind)
        du[cons] =
            0.1 *
            (
                sum(e[:, i] .* c[:, i] .* R / (1 .+ 0.5C[i] .+ 0.1R)) - m_C[i] -
                sum(c_p[i, :] .* P ./ (1 .+ theta * P .+ 0.1C[i]))
            ) *
            C[i]
    end
    # Predators dynamics.
    for (i, pred) in enumerate(p_ind)
        du[pred] =
            (sum(e_p[:, i] .* c_p[:, i] .* C / (1 + theta * P[i] .+ 0.1C)) - m_P[i]) * P[i]
    end
end

function get_R_alone(p)
    r_ind, c_ind, p_ind = get_index(p)
    u0 = zeros(richness(p))
    u0[c_ind] .= 1
    u0[p_ind] .= 1
    R_alone = zeros(length(r_ind))
    for (i, res) in enumerate(r_ind)
        u0_copy = deepcopy(u0)
        u0_copy[res] = 1
        R_alone[i] = get_sol(p; u0 = u0_copy)[res]
    end
    R_alone
end

function get_C_alone(p)
    r_ind, c_ind, p_ind = get_index(p)
    u0 = zeros(richness(p))
    u0[r_ind] .= 1
    u0[p_ind] .= 1
    C_alone = zeros(length(p_ind))
    for (i, cons) in enumerate(c_ind)
        u0_copy = deepcopy(u0)
        u0_copy[cons] = 1
        C_alone[i] = get_sol(p; u0 = u0_copy)[cons]
    end
    C_alone
end

function get_P_alone(p)
    r_ind, c_ind, p_ind = get_index(p)
    u0 = zeros(richness(p))
    u0[r_ind] .= 1
    u0[c_ind] .= 1
    P_alone = zeros(length(p_ind))
    for (i, pred) in enumerate(p_ind)
        u0_copy = deepcopy(u0)
        u0_copy[pred] = 1
        P_alone[i] = get_sol(p; u0 = u0_copy)[pred]
    end
    P_alone
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
    cp_off,
    cp_diag,
    ep_i,
    mp_i,
    theta;
    threshold = 0.1,
    maxiter = 500,
)
    u_eq = zeros(3n)
    iter = 0
    while any(u_eq .< threshold) && iter < maxiter
        r = rand(r_i, n)
        K = rand(K_i, n)
        c = rand(c_off, n, n)
        c[diagind(c)] .= c_diag
        e = rand(e_i, n, n)
        m = rand(m_i, n)
        cp = rand(cp_off, n, n)
        cp[diagind(cp)] .= cp_diag
        ep = rand(ep_i, n, n)
        mp = rand(mp_i, n)
        mr = rand(mr_i, n)
        global p = (c, e, m, r, K, mr, cp, ep, mp, theta)
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
e_i = Normal(0.1, 0)
m_i = Normal(0.1, 0)
cp_off = Uniform(0.1, 0.8)
cp_diag = 1
ep_i = e_i
mp_i = m_i
theta_list = 0.5:0.25:5
p_list = [
    generate_community(
        n,
        r_i,
        K_i,
        mr_i,
        c_off,
        c_diag,
        e_i,
        m_i,
        cp_off,
        cp_diag,
        ep_i,
        mp_i,
        theta,
    ) for theta in theta_list
]

df = DataFrame(;
    c_id = Int64[],
    sp_id = Int64[],
    sp_type = Symbol[],
    sl = Float64[],
    s = Float64[],
    tl_perturbed = Symbol[],
    theta = Float64[],
)
scenarios = [:community, :targeted]
n_rep = 5
D = LogNormal(log(0.05), 0.7)
for (i, p) in enumerate(p_list)
    r_ind, c_ind, p_ind = get_index(p)
    n_R, n_C, n_P = length.(get_index(p))
    u_eq = get_sol(p)
    R_alone = get_R_alone(p)
    C_alone = get_C_alone(p)
    P_alone = get_P_alone(p)
    SL = vcat(u_eq[r_ind] ./ R_alone, u_eq[c_ind] ./ C_alone, u_eq[p_ind] ./ P_alone)
    c, e, m, r, K, mr, cp, ep, mp, theta = deepcopy(p)
    for k in 1:n_rep, tl_perturbed in scenarios
        if tl_perturbed == :community
            global p_press = (
                c,
                e,
                m .+ rand(D, n),
                r,
                K,
                mr .+ rand(D, n),
                cp,
                ep,
                mp .+ rand(D, n),
                theta,
            )
            R_alone_press = get_R_alone(p_press)
            C_alone_press = get_C_alone(p_press)
            P_alone_press = get_P_alone(p_press)
            diff_R_alone = (R_alone_press .- R_alone) ./ R_alone
            diff_C_alone = (C_alone_press .- C_alone) ./ C_alone
            diff_P_alone = (P_alone_press .- P_alone) ./ P_alone
            diff_u_alone = vcat(diff_R_alone, diff_C_alone, diff_P_alone)
            u_press = get_sol(p_press)
            diff_u = (u_press .- u_eq) ./ u_eq
            s_press = diff_u ./ diff_u_alone
        elseif tl_perturbed == :targeted
            s_press = zeros(richness(p))
            for sp in eachindex(s_press)
                m_cat = vcat(mr, m, mp)
                m_cat[sp] += rand(D)
                mr_press, m_press, mp_press = m_cat[r_ind], m_cat[c_ind], m_cat[p_ind]
                global p_press = (c, e, m_press, r, K, mr_press, cp, ep, mp_press, theta)
                if sp in r_ind
                    idx = findfirst(==(sp), r_ind)
                    R_alone_press = get_R_alone(p_press)[idx]
                    diff_B_alone = (R_alone_press - R_alone[idx]) / R_alone[idx]
                elseif sp in c_ind
                    idx = findfirst(==(sp), c_ind)
                    C_alone_press = get_C_alone(p_press)[idx]
                    diff_B_alone = (C_alone_press - C_alone[idx]) / C_alone[idx]
                elseif sp in p_ind
                    idx = findfirst(==(sp), p_ind)
                    P_alone_press = get_P_alone(p_press)[idx]
                    diff_B_alone = (P_alone_press - P_alone[idx]) / P_alone[idx]
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
            if sp_id ∈ r_ind
                sp_type = :r
            elseif sp_id ∈ c_ind
                sp_type = :c
            elseif sp_id ∈ p_ind
                sp_type = :p
            else
                @error "Invalid index."
            end
            push!(df, (i, sp_id, sp_type, SL[sp_id], s, tl_perturbed, theta))
        end
    end
end


df_mean = combine(groupby(df, [:c_id, :sl, :sp_type, :tl_perturbed, :theta]), :s => mean)
colors = Dict(:r => :black, :c => :gray51, :p => :gray76)
inch = 96
pt = 4 / 3
cm = inch / 2.54
width = 13cm
alpha = 1
fig = Figure(; size = (width, width), fontsize = 10pt);
axes = Array{Any}(undef, 3, 2)
markersize = 5
for (j, scenario) in enumerate(scenarios)
    df_scenario = subset(df_mean, :tl_perturbed => ByRow(==(scenario)))
    for (i, type) in enumerate([:p, :c, :r])
        axes[i, j] =
            ax = Axis(
                fig[i, j];
                xlabel = "SL",
                ylabel = "Stability to press",
                yscale = Makie.pseudolog10,
            )
        color = colors[type]
        df_type = subset(df_scenario, :sp_type => ByRow(==(type)))
        scatter!(df_type.sl, 1 .- df_type.s_mean; color = df_type.theta)
    end
end
Colorbar(fig[1:3, 3]; label = "Predator interference", limits = extrema(df_mean.theta))
axes[1, 1].xlabel = ""
axes[1, 2].xlabel = ""
axes[2, 2].xlabel = ""
axes[2, 1].xlabel = ""
axes[1, 2].ylabel = ""
axes[2, 2].ylabel = ""
axes[3, 2].ylabel = ""
fig

save("figures/three-levels-theta.svg", fig)

