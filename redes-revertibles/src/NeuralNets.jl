module RNN

using Distributions

export Layer, Network, MakeNetwork, L, W, copyNetwork, forwardPropagation, backPropagation, errorGradient, BGD, SGD

logistic(s) = 1/(1+exp(-s))
logisticDerivKnown(x) = x*(1-x)
logisticDeriv(s) = logisticDerivKnown(logistic(s))

type Layer
    weights::Array{Float64,2}
end

type Network
    dimensions::Array{Int64,1}
    layers::Array{Layer,1}
end

function MakeNetwork(dimensions)
    L = length(dimensions)-1
    layers = Array{Layer,1}(L)
    for l in 1:L
        m, n = dimensions[l]+1, dimensions[l+1]
        layers[l] = Layer(rand(Uniform(-sqrt(m),sqrt(m)), (m,n)))
    end
    Network(dimensions, layers)
end

L(net) = eachindex(net.layers)
W(net,l) = net.layers[l].weights

copyNetwork(net) = deepcopy(net)

function forwardPropagation(net::Network,
                            input::Array{Float64,1})
    activations = Array{Array{Float64,1},1}(length(net.layers))
    x = [1.0; input]
    for l in L(net)
        s = W(net,l)' * x
        map!(logistic, s)
        x = [1.0; s]
        activations[l] = x
    end
    activations
end

function backPropagation(net::Network,
                         input::Array{Float64,1},
                         output::Array{Float64,1},
                         X::Array{Array{Float64,1},1})
    deltas = Array{Array{Float64,1},1}(length(net.layers))
    deltas[end] = 2*(X[end][2:end]-output).*map(logisticDerivKnown,X[end][2:end])
    
    for l in L(net)[end-1:-1:1]
        thetaDeriv = map(logisticDerivKnown, X[l][2:end])
        deltas[l] = thetaDeriv .* (W(net,l+1)*deltas[l+1])[2:end]
    end
    deltas
end

function backPropagation(net::Network,
                         input::Array{Float64,1},
                         output::Array{Float64,1})
    X = forwardPropagation(net, input)
    backPropagation(net, input, output, X)
end

function errorGradient(net::Network,
                       input::Array{Float64,1},
                       output::Array{Float64,1},
                       X::Array{Array{Float64,1},1},
                       deltas::Array{Array{Float64,1},1})
    gradEin = Array{Array{Float64,2},1}(length(net.layers))
    gradEin[1] = [1.0; input]*deltas[1]'
    for l in L(net)[2:end]
        gradEin[l] = X[l-1]*deltas[l]'
    end
    gradEin
end

function errorGradient(net::Network,
                       input::Array{Float64,1},
                       output::Array{Float64,1})
    X = forwardPropagation(net, input)
    deltas = backPropagation(net, input, output, X)
    
    errorGradient(net, input, output, X, deltas)
end

function BGD(net, inputs, outputs, learningRate, epochs)
    N = size(inputs)[2]
    errors = Array{Float64,2}(epochs, size(outputs)[1])
    grad = Array{Array{Float64,2},1}(length(net.layers))
    for l in L(net)
        grad[l] = zeros(W(net,l))
    end
    for epoch in 1:epochs
        e = zeros(outputs[:,1])
        for i in 1:N
            input  = inputs[:,i]
            output = outputs[:,i]
            
            activations = forwardPropagation(net, input)
            estimatedOutput = activations[end][2:end]
            e += (estimatedOutput-output).*(estimatedOutput-output)
            deltas = backPropagation(net, input, output, activations)
            grad += errorGradient(net, input, output, activations, deltas)
        end
        errors[epoch,:] = e/N
        for l in L(net)
            net.layers[l].weights -= learningRate .* grad[l]
            fill!(grad[l], 0.0)
        end
    end
    errors
end

function SGD(net, inputs, outputs, learningRate, epochs)
    N = size(inputs)[2]
    errors = Array{Float64,2}(epochs, size(outputs)[1])
    grad = Array{Array{Float64,2},1}(length(net.layers))
    for l in L(net)
        grad[l] = zeros(W(net,l))
    end
    for epoch in 1:epochs
        i = rand(1:N)
        input  = inputs[:,i]
        output = outputs[:,i]
        
        activations = forwardPropagation(net, input)
        estimatedOutput = activations[end][2:end]
        e = (estimatedOutput-output).*(estimatedOutput-output)
        deltas = backPropagation(net, input, output, activations)
        grad += errorGradient(net, input, output, activations, deltas)
        
        errors[epoch,:] = e/N
        for l in L(net)
            net.layers[l].weights -= learningRate .* grad[l]
            fill!(grad[l], 0.0)
        end
    end
    errors
end

function RPROP(net, inputs, outputs, Δ0, Δmax, epochs)
    ηplus = 1.2
    ηminus = 0.5
    Δmin = 1e-6
    N = size(inputs)[2]
    errors = Array{Float64,2}(epochs, size(outputs)[1])
    gradPrev = Array{Array{Float64,2},1}(length(net.layers))
    gradCurr = Array{Array{Float64,2},1}(length(net.layers))
    ΔPrev = Array{Array{Float64,2},1}(length(net.layers))
    ΔCurr = Array{Array{Float64,2},1}(length(net.layers))
    for l in L(net)
        gradPrev[l] = zeros(W(net,l))
        gradCurr[l] = zeros(W(net,l))
        ΔPrev[l] = ones(W(net,l)).*Δ0
        ΔCurr[l] = ones(W(net,l)).*Δ0
    end
    for epoch in 1:epochs
        e = zeros(outputs[:,1])
        for i in 1:N
            input  = inputs[:,i]
            output = outputs[:,i]
            
            activations = forwardPropagation(net, input)
            estimatedOutput = activations[end][2:end]
            e += (estimatedOutput-output).*(estimatedOutput-output)
            deltas = backPropagation(net, input, output, activations)
            gradCurr += errorGradient(net, input, output, activations, deltas)
        end
        errors[epoch,:] = e/N

        for l in L(net)
            w = net.layers[l].weights
            m, n = size(w)
            for i in 1:m
                for j in 1:n
                    change = gradPrev[l][i,j]*gradCurr[l][i,j]
                    if change > 0
                        ΔCurr[l][i,j] = min(ΔPrev[l][i,j]*ηplus, Δmax)
                        Δw = - sign(gradCurr[l][i,j])*ΔCurr[l][i,j]
                        w[i,j] = w[i,j] + Δw
                        gradPrev[l][i,j] = gradCurr[l][i,j]
                    elseif change < 0
                        ΔCurr[l][i,j] = max(ΔPrev[l][i,j]*ηminus, Δmin)
                        gradPrev[l][i,j] = 0
                    else
                        Δw = - sign(gradCurr[l][i,j])*ΔCurr[l][i,j]
                        w[i,j] = w[i,j] + Δw
                        gradPrev[l][i,j] = gradCurr[l][i,j]
                    end
                end
            end
            fill!(gradCurr[l], 0.0)
        end
    end
    errors
end

end
