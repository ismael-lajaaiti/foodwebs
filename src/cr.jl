using StatsBase
using GLV
using CairoMakie
using LinearAlgebra
using Distributions
using DataFrames
using DifferentialEquations
set_theme!(theme_minimal())

function consumer_resource!(du, u, p, _)
    c, e, m, r, K = p
    n_R, n_C = size(c)
    R = u[1:n_R]
    C = u[n_R+1:end]
    # Resource dynamics.
    for i in 1:n_R
        du[i] = (r[i] * (1 - R[i] / K[i]) - sum(c[i, :] .* C)) * R[i]
    end
    # Consumers dynamics.
    for i in 1:n_C
        du[i+n_R] = (sum(e[:, i] .* c[:, i] .* R) - m[i]) * C[i]
    end
end

function get_C_alone(p)
    c, e, m, r, K = p # Unpack paramters.
    C = zeros(length(m))
    for i in 1:n_C
        C[i] =
            (sum(e[:, i] .* c[:, i] .* K) - m[i]) / sum(K ./ r .* e[:, i] .* c[:, i] .^ 2)
    end
    C
end
# function get_C_alone(p)
#     c, e, m, r, K = p
#     n_r, n_c = length(m), length(r)
#     u0 = zeros(n_r + n_c)
#     C = zeros(n_c)
#     u0[1:n_r] .= 1
#     for (cons, i) in enumerate(n_r+1:n_r+n_c)
#         u0_copy = deepcopy(u0)
#         u0_copy[i] = 1
#         C[cons] = get_sol(p; u0 = u0_copy)[i]
#     end
#     C
# end
C_alone = get_C_alone(p)

function get_sol(p; tspan = (0, 1_000), u0 = fill(0.5, sum(size(p[1]))))
    prob = ODEProblem(consumer_resource!, u0, tspan, p)
    sol = DifferentialEquations.solve(prob)
    sol[end]
end

# Model parameters.
n = 10
n_C = n # Number of consumers.
n_R = n
r = fill(10.0, n_R) #rand(Uniform(15, 20), n_R)
K = fill(10.0, n_R)#rand(Uniform(10, 11), n_R)
c = rand(Uniform(0.1, 0.5), n_R, n_C)
c[diagind(c)] .= 1
e = fill(0.1, (n_R, n_C)) #rand(Uniform(0.4, 0.5), n_R, n_C)
m = fill(0.1, n_C)#rand(Uniform(0.3, 0.4), n_C)
p = (c, e, m, r, K)
u_eq = get_sol(p)
C_eq = u_eq[n_R+1:end]
C_alone = get_C_alone(p)

function generate_community(
    n,
    r_i,
    K_i,
    c_off,
    c_diag,
    e_i,
    m_i;
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
        global p = (c, e, m, r, K)
        u_eq = get_sol(p)
        iter += 1
    end
    @info iter
    p
end

r_i = Normal(10, 0)
K_i = Normal(10, 0)
c_off = Uniform(0.1, 0.2)
c_diag = 0.45
e_i = Normal(0.1, 0)
m_i = Normal(0.1, 0)
n_com = 10
p_list = []
p_list = [generate_community(n, r_i, K_i, c_off, c_diag, e_i, m_i) for _ in 1:n_com]

# All species.
n_rep = 100
D = LogNormal(log(0.05), 0.7)
sC_all = []
sR_list = []
a_all = []
for p in p_list
    global u_eq = get_sol(p)
    global C_alone = get_C_alone(p)
    c, e, m, r, K = deepcopy(p)
    sC = zeros(n_rep, n_C)
    sR = zeros(n_rep, n_C)
    kappa = zeros(n_rep, n_C)
    for k in 1:n_rep
        delta_r = rand(D, n_R)
        r_press = r .- delta_r
        p_press = (c, e, m, r_press, K)
        # delta_K = 2rand(D, n_R)
        # K_press = K .* (1 .- delta_K)
        # p_press = (c, e, m, r, K_press)
        C_alone_press = get_C_alone(p_press)
        diff_C_alone = (C_alone_press .- C_alone) ./ C_alone
        u_press = get_sol(p_press)
        diff_u = (u_press .- u_eq) ./ u_eq
        diff_R = diff_u[1:n_R]
        diff_C = diff_u[n_R+1:end]
        sC[k, :] = diff_C ./ diff_C_alone
        # sR[k, :] = diff_R ./ delta_r # delta_K
        # sR[k, :] = diff_R ./ delta_K
        kappa[k, :] = diff_C_alone
    end
    push!(a_all, mean(kappa) / harmmean(kappa))
    push!(sC_all, vec(mean(sC; dims = 1)))
    push!(sR_list, vec(mean(sR; dims = 1)))
end
a_all

# Targeted.
n_rep = 50
D = LogNormal(log(0.05), 0.7)
sC_tar = []
a_tar = []
for p in p_list
    global u_eq = get_sol(p)
    global C_alone = get_C_alone(p)
    c, e, m, r, K = deepcopy(p)
    sC = zeros(n_rep, n_C)
    a = zeros(n_rep, n_C)
    for k in 1:n_rep
        for i in 1:n_R
            delta_r = rand(D)
            r_press = deepcopy(r)
            r_press[i] -= delta_r
            p_press = (c, e, m, r_press, K)
            C_alone_press = get_C_alone(p_press)
            diff_C_alone = (C_alone_press .- C_alone) ./ C_alone
            u_press = get_sol(p_press)
            diff_u = (u_press .- u_eq) ./ u_eq
            diff_R = diff_u[1:n_R]
            diff_C = diff_u[n_R+1:end]
            sC[k, i] = diff_C[i] ./ diff_C_alone[i]
            a[k, i] = (sum(diff_C_alone) - diff_C_alone[i]) / (n_C - 1) / diff_C_alone[i]
        end
    end
    push!(a_tar, mean(a))
    push!(sC_tar, vec(mean(sC; dims = 1)))
end
a_tar

inch = 96
pt = 4 / 3
cm = inch / 2.54
width = 12cm
fig = Figure(; size = (width, 0.5width), fontsize = 10pt);
ax1 = Axis(
    fig[1, 1];
    xlabel = "SL",
    ylabel = "Consumer stability\nto press",
    title = "Community wide",
)
ax2 = Axis(fig[1, 2]; xlabel = "SL", title = "Targeted")
SL_C_list = []
for (p, sCa, sCt) in zip(p_list, sC_all, sC_tar)
    u_eq = get_sol(p)
    C_alone = get_C_alone(p)
    C_eq = u_eq[n_R+1:end]
    SL_C = C_eq ./ C_alone
    label = p == first(p_list) ? "C-R simulation" : nothing
    scatter!(ax1, SL_C, 1 .- sCa; label)
    append!(SL_C_list, SL_C)
    scatter!(ax2, SL_C, 1 .- sCt)
end
# Prediction.
sl_min, sl_max = extrema(SL_C_list)
sl_min -= 0.005
sl_values = LinRange(sl_min, sl_max, 100)
s_pred(sl, a) = 1 / sl - a * (1 / sl - 1)
lines!(
    ax1,
    sl_values,
    1 .- s_pred.(sl_values, mean(a_all));
    label = "GLV-based prediction",
    color = :black,
)
lines!(ax2, sl_values, 1 .- s_pred.(sl_values, mean(a_tar)); color = :black)
axislegend(ax1; position = :rt)
fig

save("figures/cr-glv.svg", fig)

