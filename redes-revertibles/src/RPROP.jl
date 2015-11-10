using Distributions

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

function batchGradient(net::Network,
                       inputs,
                       outputs)
    N = size(inputs)[2]
    grad = Array{Array{Float64,2},1}(length(net.layers))
    for l in L(net)
        grad[l] = zeros(W(net,l))
    end
    e = zeros(outputs[:,1])
    for i in 1:N
        input = inputs[:,i]
        output = outputs[:,i]
        activations = forwardPropagation(net, input)
        estimatedOutput = activations[end][2:end]
        diff = estimatedOutput-output
        e += diff.*diff
        grad += errorGradient(net, input, output, activations, backPropagation(net, input, output, activations))
    end
    (e/N, grad)
end
        

function RPROP(net::Network,
               inputs,
               outputs,
               epochs)
    N = size(inputs)[2]
    errors = Array{Float64,2}(epochs, size(outputs)[1])
    δEprev = Array{Array{Float64,2},1}(length(net.layers))
    Δ      = Array{Array{Float64,2},1}(length(net.layers))
    for l in L(net)
        δEprev[l] = zeros(W(net,l))
        Δ[l]      = ones(W(net,l)).*0.1
    end

    for epoch in 1:epochs
        error, δE = batchGradient(net, inputs, outputs)
        errors[epoch,:] = error
        # Update
        for l in L(net)
            w = net.layers[l].weights
            m, n = size(w)
            for i in 1:m
                for j in 1:n
                    δsign = sign(δEprev[l][i,j]*δE[l][i,j])
                    if δsign == 1
                        Δ[l][i,j] = min(Δ[l][i,j]*1.2, 50.0)
                        w[i,j] = w[i,j]-sign(δE[l][i,j])*Δ[l][i,j]
                        δEprev[l][i,j] = δE[l][i,j]
                    elseif δsign == -1
                        Δ[l][i,j] = max(Δ[l][i,j]*0.5, 1e-6)
                        δEprev[l][i,j] = 0.0
                    elseif δsign == 0
                        w[i,j] = w[i,j]-sign(δE[l][i,j])*Δ[l][i,j]
                        δEprev[l][i,j] = δE[l][i,j]
                    end
                end
            end
        end    
    end
    errors
end

function testRPROP(epochs)
    dimensions = [1,4,1]
    net = MakeNetwork(dimensions)
    inputs = collect(0:0.1:1)'
    outputs = map(x->(sin(2pi*x)+1)/2, inputs)
    N = size(inputs)[2]
    unknowns = rand(Uniform(0,1),10*N)'
    errorsRPROP = RPROP(net, inputs, outputs, epochs)
    errorsRPROP[end]
end

function testEpochs(range)
    dimensions = [1,4,1]
    net = MakeNetwork(dimensions)
    inputs = collect(0:0.1:1)'
    outputs = map(x->(sin(2pi*x)+1)/2, inputs)
    N = size(inputs)[2]
    unknowns = rand(Uniform(0,1),10*N)'
    errors = zeros(length(range))

    for i in eachindex(range)
        netCopy = copyNetwork(net)
        errorsRPROP = RPROP(netCopy, inputs, outputs, range[i])
        errors[i] = errorsRPROP[end]
    end
    errors
end
