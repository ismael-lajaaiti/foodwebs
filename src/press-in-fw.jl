using GLV
using CairoMakie
using LinearAlgebra
using Distributions
set_theme!(theme_minimal())

n_per_tl = (20, 10, 5, 5)
S = sum(n_per_tl)

function create_trophic_community(
    n_per_tl;
    D_e = LogNormal(log(0.5), 0.5),
    D_a = Normal(1, 0.1),
    D_c = Normal(-0.1, 0.01),
    u_prod = 1,
    u_cons = -0.1,
    K_i = Normal(1, 0.3),
    r_i = 1,
)
    S = sum(n_per_tl)
    A = zeros(S, S)
    tl = vcat([fill(i, n) for (i, n) in enumerate(n_per_tl)]...)
    for i in 1:S, j in (i+1):S
        if (tl[j] - tl[i]) == 1 # Predation.
            e, a = rand(D_e), rand(D_a)
            A[i, j] = -a
            A[j, i] = e * a
        elseif (tl[j] - tl[i]) == 0 # Intraguild competition.
            A[i, j] = rand(D_c)
            A[j, i] = rand(D_c)
        end
    end
    A[diagind(A)] .= -1 # Self-regulation.
    u = [t == 1 ? u_prod : u_cons for t in tl]
    K = rand(K_i, S)
    r = fill(r_i, S)
    θ = fill(1, S)
    Community(A, r, K, u, θ)
end
D_e = Normal(0.9, 0)
D_a = Normal(4 / S, 0.1 / sqrt(S))
D_c = Normal(-0.1 / S, 0.1 / sqrt(S))

iter = 0
Beq = fill(-1, S)
while any(Beq .< 0) && iter < 10_000
    c = create_trophic_community(n_per_tl; D_e, D_a, D_c)
    Beq = abundance(c)
    iter += 1
end

n_rep = 1_000
D = LogNormal(log(0.05), 0.5)
c_copy = deepcopy(c)
s_matrix = zeros(n_rep, S)
for k in 1:n_rep
    kappa = -rand(D, S)
    c_copy.K = c.K .* (1 .+ kappa)
    Bpress = abundance(c_copy)
    s_matrix[k, :] = (Bpress .- Beq) ./ Beq ./ kappa
end
s = vec(mean(s_matrix; dims = 1))
ry = relative_yield(c)
tl = vcat([fill(i, n) for (i, n) in enumerate(n_per_tl)]...)


inch = 96
pt = 4 / 3
cm = inch / 2.54
width = 8.7cm
fig = Figure(; size = (width, 0.7width), fontsize = 8pt);
ax = Axis(fig[1, 1]; xlabel = "1 / SL", ylabel = "Sensitivity to press")
for t in unique(tl)
    t_idx = tl .== t
    scatter!(1 ./ ry[t_idx], s[t_idx]; label = "TL = $t", alpha = 0.5)
end
axislegend()
fig
