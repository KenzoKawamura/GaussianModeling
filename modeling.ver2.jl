using Pkg
Pkg.activate(".")
using Comrade
using Pyehtim
using StableRNGs
using FileIO
using Dates
using Optimization
using OptimizationBBO
using Distributions, VLBIImagePriors
using DisplayAs
import CairoMakie as CM
using Pigeons
using CSV
using DataFrames
using Statistics
using Plots
using FileIO

rng = StableRNG(42)

# data setting
data_name = "a2312c.10ave.uvf.stokesI"
dir = "/Users/kenzokawamura/Dev/comrade_tutorial/comrade_2024/GitComradeTutorials/"
flag_uvdistance = 0.2e8

# model settings
Gaussian_number_min = 4
Gaussian_number_max = 4
fitting_type = "circular" # "circular" or "elliptical"

# map setting
core_position_x = μas2rad(-2500.0) # from center
core_position_y = μas2rad(-2500.0) # from center
fov_x = μas2rad(10000.0)
fov_y = μas2rad(10000.0)
number_of_pixel_x = 512
number_of_pixel_y = 512

# prior setting
core_stretch_distribution = Uniform(μas2rad(0.0), μas2rad(50.0))
component_stretch_distribution = Uniform(μas2rad(0.0), μas2rad(500.0))
component_distance_from_the_core_distribution = Uniform(0.0, μas2rad(3000.0))
component_flux_distribution = Uniform(0.0, 0.8)
component_axial_ratio_distribution = Uniform(1.0, 2.0)


obs = ehtim.obsdata.load_uvfits(joinpath(pwd(), "..", "..", "Data", dir * data_name))
obs = Pyehtim.scan_average(obs.flag_uvdist(uv_min=flag_uvdistance)).add_fractional_noise(0.02)
dlcamp, dcphase = extract_table(obs, LogClosureAmplitudes(;snrcut=3.0), ClosurePhases(;snrcut=3.0))

function modeling(component_number)

    Gaussian_number = component_number

    function model(θ, p)

        fluxes = [1.0; [θ[i] for i in 2:Gaussian_number .|> x -> Symbol("fG$x")]]
        σs = [θ[i] for i in 1:Gaussian_number .|> x -> Symbol("σG$x")]
        rs = [0.0; [θ[i] for i in 2:Gaussian_number .|> x -> Symbol("rG$x")]]
        thetas = [0.0; [θ[i] for i in 2:Gaussian_number .|> x -> Symbol("tG$x")]]

        if fitting_type == "elliptical"
            τs = [θ[i] for i in 1:Gaussian_number .|> x -> Symbol("τG$x")]
            ξs = [θ[i] for i in 1:Gaussian_number .|> x -> Symbol("ξG$x")]
        end

        sum_flux = sum(fluxes)

        if fitting_type == "circular"
        
            components = [
                (fluxes[i]/sum_flux) * shifted(stretched(Gaussian(), σs[i], σs[i]), 
                core_position_x + (rs[i]*sin(thetas[i])), core_position_y + (rs[i]*cos(thetas[i]))) 
                for i in 1:Gaussian_number
            ]
        
        elseif fitting_type == "elliptical"

            components = [
                (fluxes[i]/sum_flux) * shifted(rotated(stretched(Gaussian(), σs[i], σs[i] * (τs[i])), ξs[i]),
                core_position_x + (rs[i]*sin(thetas[i])), core_position_y + (rs[i]*cos(thetas[i])))
                for i in 1:Gaussian_number
            ]

        else

            println("fitting_type is invalid!")

        end

        return sum(components)
    end

    filename = data_name * "." * string(Gaussian_number) * "." * fitting_type
    mkpath(dir * "data/" * filename)

    prior = (
        σG1 = core_stretch_distribution, 
    )

    if (Gaussian_number > 1)
        for i in 2:Gaussian_number
            prior = merge(prior, (
                Symbol("fG$i") => component_flux_distribution,
                Symbol("rG$i") => component_distance_from_the_core_distribution,
                Symbol("tG$i") => Uniform(0.0, 2π),
                Symbol("σG$i") => component_stretch_distribution
            ))
        end
    end

    if fitting_type == "elliptical"

        for i in 1:Gaussian_number
            prior = merge(prior, (
                Symbol("τG$i") => component_axial_ratio_distribution,
                Symbol("ξG$i") => Uniform(0.0, 1π)
            ))
        end
    end

    df_prior = DataFrame(prior)

    skym = SkyModel(model, prior, imagepixels(fov_x, fov_y, number_of_pixel_x, number_of_pixel_y))

    post = VLBIPosterior(skym, dlcamp, dcphase)

    cpost = ascube(post)

    fpost = asflat(post)

    xopt, sol = comrade_opt(post, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters=50_000);

    g = imagepixels(fov_x, fov_y, number_of_pixel_x, number_of_pixel_y)
    intensity = intensitymap(skymodel(post, xopt), g)
    fig = imageviz(intensity, colormap=:afmhot, size=(500, 400));
    DisplayAs.Text(DisplayAs.PNG(fig))

    pt = pigeons(target=cpost, explorer=SliceSampler(), record=[traces, round_trip, log_sum_ratio], n_chains=32, n_rounds=8, multithreaded = true)

    chain = sample_array(cpost, pt)

    df = DataFrame(chain)

    imgs = intensitymap.(skymodel.(Ref(post), sample(chain, 100)), Ref(g))
    fig = imageviz(imgs[end], colormap=:afmhot)

    meanimg = mean(imgs)
    ComradeBase.save(dir * "data/" * filename * ".mean_image.fits", meanimg)
    fig = imageviz(meanimg, colormap=:afmhot);
    save(dir * "data/" * filename * ".mean_image.png", DisplayAs.PNG(fig))

    p1 = Plots.plot(dlcamp);
    p2 = Plots.plot(dcphase);
    uva = uvdist.(datatable(dlcamp))
    uvp = uvdist.(datatable(dcphase))
    for i in 1:10
        mobs = simulate_observation(post, sample(chain, 1)[1])
        mlca = mobs[1]
        mcp  = mobs[2]
        Plots.scatter!(p1, uva, mlca[:measurement], color=:grey, label=:none, alpha=0.2)
        Plots.scatter!(p2, uvp, atan.(sin.(mcp[:measurement]), cos.(mcp[:measurement])), color=:grey, label=:none, alpha=0.2)
    end
    p = Plots.plot(p1, p2, layout=(2,1));

    p = residual(post, chain[end]);
    save(dir * "data/" * filename * ".residual.png", DisplayAs.PNG(p))

    c2_cphase, c2_clamp = chi2(post, chain[end])
    c2_cphase /= length(dcphase)
    c2_clamp /= length(dlcamp)
    @info "CP  Chi2 2: $(c2_cphase)"    
    @info "LCA Chi2 2: $(c2_clamp)"

    return df, filename
end

function parse_parameters(row)

    parameters = split(strip(string(row[1]), ['(', ')']), ", ")

    parsed = Dict{String, Any}()
    
    for param in parameters
        key, value = split(param, " = ")
        parsed[key] = parse(Float64, value)
    end
    
    return parsed
end

function parameter_extract(row::DataFrameRow, param::String, func::Union{Function, Nothing} = nothing, initial_value::Union{Float64, Nothing} = nothing)
    # 指定された param で始まる列を選択
    columns_df = select(DataFrame(row), r"^" * param)
    
    # func が指定されていれば、それを適用
    if func != nothing
        columns_df .= columns_df .* func(1.0)
    end
    
    # initial_value が指定されていれば、最初の列にその値を代入
    if initial_value != nothing
        columns_df[!, Symbol(param * "1")] = [initial_value]
    end
    
    return columns_df
end

function parameter_replace(sorted_numbers::Vector{Any}, param::String, parameter_columns::DataFrame)
    sorted_df = DataFrame()
    for i in 1:length(sorted_numbers)
        # 動的に列名を作成して値を代入
        sorted_df[!, Symbol(param * string(i))] = parameter_columns[!, Symbol(param * string(sorted_numbers[i]))]
    end
    return sorted_df
end

function chains_summary(parsed_df::DataFrame, fitting_type::String)
    MCMCchains = DataFrame()
    for i in 1:nrow(parsed_df)
        df_row = parsed_df[i, :]
        rG_columns_df = parameter_extract(df_row, "rG", rad2μas, 0.0)
        column_names = names(rG_columns_df)
        values = collect(rG_columns_df[1, :])
        sorted_values = sort(values)
        index = []
        for j in 1:length(values)
            sorted_number = findfirst(x -> x == sorted_values[j], values)
            push!(index, sorted_number)
        end
        sorted_columns = []
        for j in 1:length(values)
            push!(sorted_columns, column_names[index[j]])
        end
        sorted_numbers = []
        for j in 1:length(sorted_columns)
            numbers_only = filter(isdigit, sorted_columns[j])
            push!(sorted_numbers, numbers_only)
        end
        rG_sorted_df = parameter_replace(sorted_numbers, "rG", parameter_extract(df_row, "rG", rad2μas, 0.0))
        tG_sorted_df = parameter_replace(sorted_numbers, "tG", parameter_extract(df_row, "tG", rad2deg, 0.0))
        σG_sorted_df = parameter_replace(sorted_numbers, "σG", parameter_extract(df_row, "σG", rad2μas,))
        fG_sorted_df = parameter_replace(sorted_numbers, "fG", parameter_extract(df_row, "fG",nothing,1.0))
        allparameters = hcat(rG_sorted_df, tG_sorted_df, σG_sorted_df, fG_sorted_df)
        if fitting_type == "elliptical"
            τG_sorted_df = parameter_replace(sorted_numbers, "τG", parameter_extract(df_row, "τG", rad2deg,))
            ξG_sorted_df = parameter_replace(sorted_numbers, "ξG", parameter_extract(df_row, "ξG", nothing,))
            allparameters = hcat(allparameters, τG_sorted_df, ξG_sorted_df)
        end
        append!(MCMCchains, allparameters)
    end
    return MCMCchains
end

for i in Gaussian_number_min:Gaussian_number_max
    @time begin
        println(string(i, " components modeling begins"))
        sdf, fname = modeling(i)
        parsed_data = [parse_parameters(row) for row in eachrow(sdf)]
        parsed_df = DataFrame(parsed_data)
        chains_df = chains_summary(parsed_df, fitting_type)
        CSV.write(dir * "data/" * fname * ".MCMCchains.csv", chains_df)
        parameters = names(chains_df)
        means = [mean(chains_df[!, col]) for col in names(chains_df)]
        std_devs = [std(chains_df[!, col]) for col in names(chains_df)]
        summary_df = DataFrame(Parameter = parameters, 
                            Mean = means, 
                            StdDev = std_devs)
        println(summary_df)
        CSV.write(dir * "data/" * fname * ".MCMCsummary.csv", summary_df)
        @info "report was saved as ./$(fname).~"
    end
end