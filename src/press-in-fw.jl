using StatsBase
using GLV
using CairoMakie
using LinearAlgebra
using Distributions
using DataFrames
set_theme!(theme_minimal())

n_per_tl = (10, 10, 10, 10)
S = sum(n_per_tl)
n_tl = length(n_per_tl)

function create_trophic_community(
    n_per_tl;
    D_e = LogNormal(log(0.5), 0.5),
    D_a = Normal(1, 0.1),
    D_c = Normal(-0.1, 0.01),
    K_i = Normal(1, 0.3),
    Z = 1,
    r_i = 1,
)
    S = sum(n_per_tl)
    A = zeros(S, S)
    tl = vcat([fill(i, n) for (i, n) in enumerate(n_per_tl)]...)
    m = Z .^ (tl .- 1) # Metabolism.
    for i in 1:S, j in (i+1):S
        if (tl[j] - tl[i]) == 1 # Predation.
            e, a = rand(D_e), rand(D_a)
            A[i, j] = -m[j] / m[i] * a
            A[j, i] = e * a
        elseif (tl[j] - tl[i]) == 0 # Intraguild competition.
            A[i, j] = rand(D_c)
            A[j, i] = rand(D_c)
        end
    end
    A[diagind(A)] .= -1 # Self-regulation.
    u = [t == 1 ? 1 : -0.1 for t in tl] #.* m
    K = rand(K_i, S)
    r = fill(r_i, S) .* m
    θ = fill(1, S)
    Community(A, r, K, u, θ)
end

params_dict = Dict(
    :inversed => (e = 0.8, Z = 0.1, a = 10),
    :pyramid => (e = 0.8, Z = 0.8, a = 4),
    :cascade => (e = 0.8, Z = 0.8, a = 15),
)

com_dict = Dict()
for (key, params) in params_dict
    D_e = Normal(params.e, 0)
    D_a = Normal(params.a / S, 0 / sqrt(S))
    D_c = Normal(-0.0 / S, 0.0 / sqrt(S))
    iter = 0
    Beq = fill(-1, S)
    while any(Beq .< 1e-2) && iter < 10_000
        global c = create_trophic_community(n_per_tl; D_e, D_a, D_c, Z = params.Z)
        Beq = abundance(c)
        iter += 1
    end
    @info all(Beq .> 0)
    com_dict[key] = c
end

df = DataFrame(;
    com = Symbol[],
    tl = Float64[],
    sl = Float64[],
    s = Float64[],
    B = Float64[],
)
n_rep = 10_000
D = LogNormal(log(0.01), 0.5)
tl = vcat([fill(i, n) for (i, n) in enumerate(n_per_tl)]...)
for (key, c) in com_dict
    ry = relative_yield(c)
    Beq = abundance(c)
    c_copy = deepcopy(c)
    for k in 1:n_rep
        # kappa = -0.1 * rand(D, S)
        # c_copy.K = c.K .* (1 .+ kappa)
        du_u = -0.1 * rand(D, S)
        u_press = c.u + du_u .* abs.(c.u)
        c_copy.u = u_press
        Bpress = abundance(c_copy)
        s = (Bpress .- Beq) ./ Beq ./ du_u
        append!(df, (com = fill(key, S), tl = tl, sl = ry, s = s, B = Beq))
    end
end
df_avg = combine(groupby(df, [:com, :tl, :sl, :B]), :s => mean)

function predict_sl(e, Z, a, B_per_tl, S_per_tl)
    S = sum(S_per_tl)
    n_tl = length(B_per_tl)
    pred = zeros(n_tl)
    for t in 1:n_tl
        B_mean = B_per_tl[t] / S_per_tl[t]
        B_prey = t > 1 ? B_per_tl[t-1] : 0
        B_pred = t < n_tl ? B_per_tl[t+1] : 0
        pred[t] = 1 / (1 - (e * a / S * B_prey - Z * a / S * B_pred) / B_mean)
    end
    pred
end

function predict_s(sl, D_press; v = 1)
    press_vals = rand(D_press, 100_000)
    p = mean(press_vals) / harmmean(press_vals)
    v / sl * (1 - p) + p
end

inch = 96
pt = 4 / 3
cm = inch / 2.54
width = 10cm
fig = Figure(; size = (width, width), fontsize = 8pt);
for (i, com) in enumerate(keys(com_dict))
    df_com = subset(df_avg, :com => ByRow(==(com)))
    ax1 = Axis(fig[i, 1]; xlabel = "Trophic level", ylabel = "SL")
    boxplot!(df_com.tl, df_com.sl; color = :grey)
    ax2 = Axis(fig[i, 2]; xlabel = "Trophic level", ylabel = "Sensitivity to press")
    boxplot!(df_com.tl, df_com.s_mean; color = :grey)
    ax3 = Axis(fig[i, 3]; xlabel = "Total biomass", ylabel = "Trophic level")
    tl_unique = sort(unique(df_com.tl))
    B_tl = [mean(df_com.B[df_com.tl.==t]) for t in tl_unique]
    p = params_dict[com]
    sl_pred = predict_sl(p.e, p.Z, p.a, B_tl, n_per_tl)
    s_pred = predict_s.(sl_pred, D)
    barplot!(tl_unique, B_tl; color = :grey, direction = :x)
    scatter!(ax1, tl_unique, sl_pred; color = :red)
    scatter!(ax2, tl_unique, s_pred; color = :red)
    Label(fig[i, 4], string(com); rotation = 3pi / 2, tellheight = false)
end
fig

save("figures/tl-vs-sl.png", fig)

inch = 96
pt = 4 / 3
cm = inch / 2.54
width = 10cm
fig = Figure(; size = (width, width), fontsize = 8pt);
for (i, com) in enumerate(keys(com_dict))
    df_com = subset(df_avg, :com => ByRow(==(com)), :tl => ByRow(>=(2)))
    ax1 = Axis(fig[i, 1]; xlabel = "Trophic level", ylabel = "SL")
    boxplot!(df_com.tl, df_com.sl; color = :grey)
    ax2 = Axis(fig[i, 2]; xlabel = "Trophic level", ylabel = "Sensitivity to press")
    boxplot!(df_com.tl, df_com.s_mean; color = :grey)
    ax3 = Axis(fig[i, 3]; xlabel = "Total biomass", ylabel = "Trophic level")
    df_com = subset(df_avg, :com => ByRow(==(com)))
    tl_unique = sort(unique(df_com.tl))
    B_tl = [mean(df_com.B[df_com.tl.==t]) for t in tl_unique]
    p = params_dict[com]
    sl_pred = predict_sl(p.e, p.Z, p.a, B_tl, n_per_tl)
    s_pred = predict_s.(sl_pred, D)
    # scatter!(ax2, tl_unique[2:end], s_pred[2:end]; color = :red)
    barplot!(tl_unique, B_tl; color = :grey, direction = :x)
    scatter!(ax1, tl_unique[2:end], sl_pred[2:end]; color = :red)
    Label(fig[i, 4], string(com); rotation = 3pi / 2, tellheight = false)
end
fig

inch = 96
pt = 4 / 3
cm = inch / 2.54
width = 15cm
color_dict = Dict(1 => :black, 2 => :gray47, 3 => :grey66, 4 => :gray80)
fig = Figure(; size = (width, width), fontsize = 10pt);
for (i, com) in enumerate(keys(com_dict))
    ax = Axis(fig[0, i]; xlabel = "Total biomass", ylabel = "Trophic level")
    df_com = subset(df_avg, :com => ByRow(==(com)))
    tl_unique = sort(unique(df_com.tl))
    B_tl = [mean(df_com.B[df_com.tl.==t]) for t in tl_unique]
    p = params_dict[com]
    barplot!(tl_unique, B_tl; direction = :x, color = [color_dict[i] for i in 1:n_tl])
    for (j, df_tl) in enumerate(groupby(df_com, :tl))
        tl = df_tl.tl |> first
        color = color_dict[tl]
        xlabel = j == 1 ? "1 / SL" : ""
        ylabel = i == 1 ? "Sensitivity" : ""
        ax = Axis(fig[n_tl-j+1, i]; xlabel, ylabel)
        scatter!(1 ./ df_tl.sl, df_tl.s_mean; color)
    end
end
fig


inch = 96
pt = 4 / 3
cm = inch / 2.54
width = 13cm
fig = Figure(; size = (0.5width, width), fontsize = 10pt);
for (i, com) in enumerate(keys(com_dict))
    df_com = subset(df_avg, :com => ByRow(==(com)))
    ax = Axis(
        fig[i, 1];
        xlabel = "SL",
        ylabel = "Sensitivity",
        yscale = Makie.pseudolog10,
        xscale = Makie.pseudolog10,
    )
    for df_tl in groupby(df_com, :tl)
        scatter!(1 ./ df_tl.sl, df_tl.s_mean)
    end
end
fig

save("figures/tl-vs-sl_no-producers.png", fig)
