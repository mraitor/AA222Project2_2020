#=
        project2.jl -- This is where the magic happens!

    All of your code must either live in this file, or be `include`d here.
=#
using LinearAlgebra
using Distributions

"""
    optimize(f, g, c, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `c`: Constraint function for 'f'
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
function optimize(f, g, c, x0, n, prob)
    x_best = x0
#     dim_c_out = length(c(x0))
    if prob == "simple2"
        mu = x0
        scale = 5.0
        stdM = Matrix(scale*I, length(x0), length(x0))
        D = MvNormal(mu, stdM)
        x_best = mix_cross(f, g, c, x0, D, n, prob)
#         println("end optimize " ,count(f, g) + count(c))
    elseif prob == "simple3"
        mu = x0
        scale = 0.88
        stdM = Matrix(scale*I, length(x0), length(x0))
        D = MvNormal(mu, stdM)
        x_best = count_cross(f, g, c, x0, D, n, prob)
#         println("end optimize " ,count(f, g) + count(c))
    elseif prob == "secret1"
        mu = x0
        scale = 0.1 #5 = 440
        stdM = Matrix(scale*I, length(x0), length(x0))
        D = MvNormal(mu, stdM)
        x_best = count_cross(f, g, c, x0, D, n, prob)
#         println("end optimize " ,count(f, g) + count(c))
    elseif prob == "secret2"
        mu = x0
        scale = 100000.0
        stdM = Matrix(scale*I, length(x0), length(x0))
        D = MvNormal(mu, stdM)
        x_best = mix_cross(f, g, c, x0, D, n, prob)
#         println("end optimize " ,count(f, g) + count(c))
    else #prob == "simple1"
        mu = x0
        scale = 0.99
        stdM = Matrix(scale*I, length(x0), length(x0))
        D = MvNormal(mu, stdM)
        x_best = mix_cross(f, g, c, x0, D, n, prob)
#         println("end optimize " ,x_best)
    end
    
    return x_best
end

function optimize_history(f, g, c, x0, n, prob)
    x_history = Array{Array{Float64,1}}(undef, 1);
    x_history[1] = x0
#     dim_c_out = length(c(x0))
    if prob == "simple2"
        mu = x0
        scale = 5.0
        stdM = Matrix(scale*I, length(x0), length(x0))
        D = MvNormal(mu, stdM)
        x_history = count_cross_history(f, g, c, x0, D, n, prob)
#         println("end optimize " ,count(f, g) + count(c))
    elseif prob == "simple3"
        mu = x0
        scale = 0.88
        stdM = Matrix(scale*I, length(x0), length(x0))
        D = MvNormal(mu, stdM)
        x_history = count_cross(f, g, c, x0, D, n, prob)
#         println("end optimize " ,count(f, g) + count(c))
    elseif prob == "secret1"
        mu = x0
        scale = 0.1 #5 = 440
        stdM = Matrix(scale*I, length(x0), length(x0))
        D = MvNormal(mu, stdM)
        x_history = count_cross(f, g, c, x0, D, n, prob)
#         println("end optimize " ,count(f, g) + count(c))
    elseif prob == "secret2"
        mu = x0
        scale = 100000.0
        stdM = Matrix(scale*I, length(x0), length(x0))
        D = MvNormal(mu, stdM)
        x_history = mix_cross(f, g, c, x0, D, n, prob)
#         println("end optimize " ,count(f, g) + count(c))
    else #prob == "simple1"
        mu = x0
        scale = 0.99
        stdM = Matrix(scale*I, length(x0), length(x0))
        D = MvNormal(mu, stdM)
        x_history = count_cross_history(f, g, c, x0, D, n, prob)
#         println("end optimize " ,x_best)
    end
    
    return x_history
end

#use count penalty and cross entropy to solve constrained minimization problem
#f, g, c, x0, D, n, prob,
# ρ: initial penalty
# γ: multiplying penalty (with iteration)
function count_cross(f, g, c, x0, D, n, prob)
    x_best = x0
    if prob == "simple1"
        m = 100
        m_elite = 5
        ρ=10 
        γ=3
    elseif prob == "simple2"
        m = 100
        m_elite = 10
        ρ= 10
        γ=2
    elseif prob == "simple3"
        m = 200
        m_elite = 10
        ρ= 10
        γ=3
    elseif prob == "secret1"
        m = 500
        m_elite = 100
        ρ=20
        γ=3
    else
        m = 200
        m_elite = 12
        ρ=1000
        γ=2
    end
    #TODO calc new cost funciton, p, from f, c, and x0 (as initial mean)
#     for k in 1:((n/m)-1) #every iteration, uses m function evaluations
#     k = 0
    while (count(f, g) + count(c)) < (n-(2*m)) #multiply by 2 because every eval of f also evaluates c, so "double counts" nevals
        p = x -> sum( x-> x>0, c(x) )
        pk = x -> f(x) + ρ*p(x)
        D = generatee_distribution(pk, D, m, m_elite)
        x_best = mean(D)#TODO extract mu from distribution
        ρ *= γ
        if p(x_best) == 0
#             scatter!( [x_best[1]], [x_best[1]])
#             println("feasible, return early")
            return x_best
        end
#         println("current count")
#         println(count(f, g) + count(c))
    end
#     scatter!( [x_best[1]], [x_best[1]])
#        println("end count_cross",count(f, g) + count(c))
#     display((n/(2*m))
#     println("break at end")
    return x_best 
end

#mixed constraint penalty method
function mix_cross(f, g, c, x0, D, n, prob)
    x_best = x0
     ρ_quad = 1
    if prob == "simple1"
        m = 100
        m_elite = 5
        ρ_count=10 
        γ=3
    elseif prob == "simple2"
        m = 100
        m_elite = 10
        ρ_count= 10
        γ=2
    elseif prob == "simple3"
        m = 200
        m_elite = 10
        ρ_count= 10
        γ=3
    elseif prob == "secret1"
        m = 500
        m_elite = 100
        ρ_count=20
        γ=3
    else
        m = 1000
        m_elite = 14
        ρ_count=1
        γ=2
        ρ_quad = 1
    end
    
   
    
    #TODO calc new cost funciton, p, from f, c, and x0 (as initial mean)
#     for k in 1:((n/m)-1) #every iteration, uses m function evaluations
#     k = 0
    while (count(f, g) + count(c)) < (n-(2*m)) #multiply by 2 because every eval of f also evaluates c, so "double counts" nevals
#         p_count = x -> sum( x-> x>0, c(x) )
# #         p_quad = x->sum( (max(c(x),0)).^2
#         pk = x -> f(x) + ρ*p_count(x) + p_quadratic(c,x)
        pk = x -> f(x) + p_mix(c, x, ρ_count, ρ_quad)
        D = generatee_distribution(pk, D, m, m_elite)
        x_best = mean(D)#TODO extract mu from distribution
        ρ_count *= γ
        ρ_quad *= γ
        if p_mix(c, x_best, ρ_count, ρ_quad) == 0
#             scatter!( [x_best[1]], [x_best[1]])
#             println("feasible, return early")
#             x_best = x_tmp
            return x_best
        end
#         println("current count")
#         println(count(f, g) + count(c))
    end
#     scatter!( [x_best[1]], [x_best[1]])
#     "end count_cross",count(f, g) + count(c), 
#     display((n/(2*m))
#     println("break at end")
    return x_best 
end

function count_cross_history(f, g, c, x0, D, n, prob)
    x_best = x0
    x_history = Array{Array{Float64,1}}(undef, 1);
    x_history[1] = x0
    if prob == "simple1"
        m = 100
        m_elite = 5
        ρ=10 
        γ=3
    elseif prob == "simple2"
        m = 100
        m_elite = 10
        ρ= 10
        γ=2
    elseif prob == "simple3"
        m = 200
        m_elite = 10
        ρ= 10
        γ=3
    elseif prob == "secret1"
        m = 500
        m_elite = 100
        ρ=20
        γ=3
    else
        m = 200
        m_elite = 12
        ρ=1000
        γ=2
    end
    #TODO calc new cost funciton, p, from f, c, and x0 (as initial mean)
#     for k in 1:((n/m)-1) #every iteration, uses m function evaluations
#     k = 0
    while (count(f, g) + count(c)) < (n-(2*m)) #multiply by 2 because every eval of f also evaluates c, so "double counts" nevals
        p = x -> sum( x-> x>0, c(x) )
        pk = x -> f(x) + ρ*p(x)
        D = generatee_distribution(pk, D, m, m_elite)
        x_best = mean(D)#TODO extract mu from distribution
        ρ *= γ
#         if p(x_best) == 0
# #             scatter!( [x_best[1]], [x_best[1]])
# #             println("feasible, return early")
#             x_history = push!(x_history, x_best);
#             return x_history
#         end
#         println("current count")
#         println(count(f, g) + count(c))
        x_history = push!(x_history, x_best);
    end
#     scatter!( [x_best[1]], [x_best[1]])
#        println("end count_cross",count(f, g) + count(c))
#     display((n/(2*m))
#     println("break at end")
    return x_history 
end


#mixed constraint penalty method using cross-entropy, returns mean of each distribution generated
function mix_cross_history(f, g, c, x0, D, n, prob)
    x_best = x0
    x_history = Array{Array{Float64,1}}(undef, 1);
    x_history[1] = x0
     ρ_quad = 1
    if prob == "simple1"
        m = 100
        m_elite = 5
        ρ_count=10 
        γ=3
#         println("simple1 params set")
    elseif prob == "simple2"
        m = 100
        m_elite = 10
        ρ_count= 10
        γ=2
    elseif prob == "simple3"
        m = 200
        m_elite = 10
        ρ_count= 10
        γ=3
    elseif prob == "secret1"
        m = 500
        m_elite = 100
        ρ_count=20
        γ=3
    else
        m = 1000
        m_elite = 14
        ρ_count=1
        γ=2
        ρ_quad = 1
    end
    
   
    
    #TODO calc new cost funciton, p, from f, c, and x0 (as initial mean)
#     for k in 1:((n/m)-1) #every iteration, uses m function evaluations
#     k = 0
    while (count(f, g) + count(c)) < (n-(2*m)) #multiply by 2 because every eval of f also evaluates c, so "double counts" nevals
#         p_count = x -> sum( x-> x>0, c(x) )
# #         p_quad = x->sum( (max(c(x),0)).^2
#         pk = x -> f(x) + ρ*p_count(x) + p_quadratic(c,x)
        pk = x -> f(x) + p_mix(c, x, ρ_count, ρ_quad)
        D = generatee_distribution(pk, D, m, m_elite)
        x_best = mean(D)# extract mu from distribution
#         println("x_best = ")
#         println(x_best)
        ρ_count *= γ
        ρ_quad *= γ
#         if p_mix(c, x_best, ρ_count, ρ_quad) == 0
# #             scatter!( [x_best[1]], [x_best[1]])
# #             println("feasible, return early")
# #             
# #             x_best
#             x_history = push!(x_history, x_best);
#             return x_history
#         end
#         println("current count")
#         println(count(f, g) + count(c))
        
        x_history = push!(x_history, x_best);
    end
#     scatter!( [x_best[1]], [x_best[1]])
#     "end count_cross",count(f, g) + count(c), 
#     display((n/(2*m))
#     println("break at end")
    
    return x_history 
end
 
#Kochenderfer, Mykel J.. Algorithms for Optimization (The MIT Press) (p. 135). The MIT Press.
#geneerate new distribution based on m_eelite samples from m samples from current distribution
# f is cost function, P is distribution, m is num. samples, m_elite is num elite smaples
function generatee_distribution(f, P, m, m_elite)
    samples = rand(P, m)
    order = sortperm([f(samples[:,i]) for i in 1:m])
    P = fit(typeof(P), samples[:,order[1:m_elite]])
    return P
end 

function p_mix(c, x, ρ_count, ρ_quad)
#     p_count = x -> sum( x-> x>0, c(x) )
#         p_quad = x->sum( (max(c(x),0)).^2
#     pk = x -> f(x) + ρ*p_count(x) + p_quadratic(c,x)
    penalty = 0
    count_penalty = 0
    quadratic_penalty = 0
    c_x = c(x)
     #count penalty
    count_penalty = sum( y -> y > 0, c_x )
    for i in 1:length(x)
#         c_i = c(x)
        quadratic_penalty = quadratic_penalty + maximum([c_x[i], 0])^2
    end
    penalty = ρ_count*count_penalty + ρ_quad*quadratic_penalty
    return penalty
end

function penalty_method(f, p, x, k_max; ρ=1, γ=2)
    for k in 1: k_max
        x = minimize(x -> f(x) + ρ*p(x), x)
        ρ *= γ
        if p(x) == 0
            return x 
        end
    end
    return x
end 

function minimize(f, g, x0, n, prob)
    #     Basing first attempt on Nesterov's momentum gradient descent as described in Alg. 5.4 in the course textbook
 	v = vec(zeros(length(x0),1)); #set momentum to zero
	i = 1;   
    x_history = Array{Array{Float64,1}}(undef, 1);
 	x_history[1] = x0; 
    alpha = 0.9; #learning rate
    beta = 0.8;#momentum decay, 0 = pure gradient method  
    
    #augmentes lagrange setup
    λ = zeros(length(h(x)))
    
    while count(f, g) < n
        xi_vec = vec(x_history[i]); 
        tmp_vec = xi_vec + beta*v;
        v = beta*v - (alpha^i)*g(tmp_vec); #adjusted Nestov's alg. to incorporate decreasing learning rate
        normThresh = 1;
        if norm(v) > normThresh
            v = v / norm(v)
        end
        xpi_vec = xi_vec + v;
#         x_history = hcat(x_history, xpi_vec);
         x_history = push!(x_history, xpi_vec);
#         println(typeof(x_history))
        i = i + 1;
    end

    return vec(x_history[i-1])
end

#Kochenderfer, Mykel J.. Algorithms for Optimization (The MIT Press) (p. 183).
function augmented_lagrange_method(f, h, x, k_max; ρ=1, γ=2)
    λ = zeros(length(h(x)))
    for k in 1: k_max
        p = x -> f(x) + ρ/2*sum(h(x).^2) - λ⋅h(x)
        x = minimize(x -> f(x) + p(x), x)
        ρ *= γ
        λ -= ρ*h(x)
    end
    return x
end 


