using GLV
using CairoMakie
using LinearAlgebra
using Distributions
set_theme!(theme_minimal())

n_per_tl = (20, 10, 10, 5, 2)
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
D_e = Normal(1, 0)
D_a = Normal(4 / S, 0.1 / sqrt(S))
D_c = Normal(-0.1 / S, 0.1 / sqrt(S))

iter = 0
Beq = fill(-1, S)
while any(Beq .< 0) && iter < 10_000
    c = create_trophic_community(n_per_tl; D_e, D_a, D_c)
    Beq = abundance(c)
    iter += 1
end

alpha = 1
n_rep = 1_000
D = LogNormal(log(0.05), 0.5)
tspan = (0, 10_000)
noise_intensity = fill(0.01, S)
noise!(du, u, p, t) =
    for i in eachindex(du)
        du[i] = noise_intensity[i] * Beq[i]^alpha
    end
sol = solve(c, Beq, tspan, noise!)
cv = sqrt.(var(sol; dims = 2)) ./ noise_intensity ./ Beq |> vec
ry = relative_yield(c)

inch = 96
pt = 4 / 3
cm = inch / 2.54
width = 8.7cm
fig = Figure(; size = (width, 0.7width), fontsize = 8pt);
tl = vcat([fill(i, n) for (i, n) in enumerate(n_per_tl)]...)
ax = Axis(fig[1, 1]; xlabel = "Beq", ylabel = "CV", yscale = log10, xscale = log10)
for t in unique(tl)
    t_idx = tl .== t
    scatter!(Beq[t_idx], cv[t_idx]; label = "TL = $t")
end
axislegend()
fig

inch = 96
pt = 4 / 3
cm = inch / 2.54
width = 17.8cm
fig = Figure(; size = (width, 0.4width), fontsize = 8pt);
tl = vcat([fill(i, n) for (i, n) in enumerate(n_per_tl)]...)
ax1 = Axis(
    fig[1, 1];
    xlabel = "SL",
    ylabel = "Temporal stability",
    yscale = log10,
    xscale = log10,
)
ax2 = Axis(fig[1, 2]; xlabel = "B", ylabel = "", yscale = log10, xscale = log10)
for t in unique(tl)
    t_idx = tl .== t
    scatter!(
        ax1,
        sqrt.(abs.(ry[t_idx])),
        1 ./ (cv[t_idx] .* sqrt.(abs.(c.u[t_idx])));
        label = "TL = $t",
    )
    scatter!(ax2, Beq[t_idx], 1 ./ (cv[t_idx] .* sqrt.(abs.(c.u[t_idx]))))
end
axislegend(ax1; position = :rb)
fig

inch = 96
pt = 4 / 3
cm = inch / 2.54
width = 8.7cm
fig = Figure(; size = (width, 0.7width), fontsize = 8pt);
tl = vcat([fill(i, n) for (i, n) in enumerate(n_per_tl)]...)
basal_sp = tl .== 2
ax = Axis(fig[1, 1]; xlabel = "CV true", ylabel = "CV prediction")
cv_pred(B, ry, u, alpha) = 1 / sqrt(2) * B^(alpha - 1) / sqrt(abs(u * ry))
ry_val = LinRange(0.1, 0.7, 100)
scatter!(cv[basal_sp], cv_pred.(Beq[basal_sp], ry[basal_sp], c.u[basal_sp], alpha))
ablines!(0, 1)
# axislegend(; position = :rb)
fig


plot(sol; color = tl)

s = vec(mean(s_matrix; dims = 1))
