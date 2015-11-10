include("NeuralNets.jl")

using Gadfly
using Colors
using Distributions

function testBGDvsSGD()
    let dimensions = [1, 4, 1],
        net = RNN.MakeNetwork(dimensions),
        inputs  = collect(0:0.1:1)',
        outputs = map(x->(sin(2pi*x)+1)/2, inputs),
        N = size(inputs)[2],
        epochs = 50000
        
        unknowns = rand(Uniform(0,1),10*N)'
        
        netBGD = RNN.copyNetwork(net)
        tic()
        errorsBGD = RNN.BGD(netBGD, inputs, outputs, 0.4, epochs)
        timeBGD = toq()
        estimationBGD = [RNN.forwardPropagation(netBGD, unknowns[:,i])[end][2] for i in 1:10*N]'
        
        netSGD = RNN.copyNetwork(net)
        tic()
        errorsSGD = RNN.SGD(netSGD, inputs, outputs, 5.0, epochs)
        timeSGD = toq()
        estimationSGD = [RNN.forwardPropagation(netSGD, unknowns[:,i])[end][2] for i in 1:10*N]'
        
        p = plot(
                 layer(x=inputs, y=outputs, Geom.line, Theme(default_color = RGB(0,0,0))),
                 layer(x=unknowns, y=estimationBGD, Geom.point, Theme(default_color = RGB(1,0,0))),
                 layer(x=unknowns, y=estimationSGD, Geom.point, Theme(default_color = RGB(0,1,0))),
                 Guide.title("Comparación de métodos de entrenamiento"),
                 Guide.xlabel("x"),
        Guide.ylabel("(sin(2πx)+1)/2"),
        Guide.manual_color_key("", ["objetivo", "hipótesis BGD (0.01)", "hipótesis SGD (0.4)"], [RGB(0,0,0),RGB(1,0,0),RGB(0,1,0)])
        )
        
        
        println("BGD last error         $(errorsBGD[end])")
        println("SGD last error         $(errorsSGD[end])")
        println("--------------")
        println("BGD training duration  $(timeBGD) seconds")
        println("SGD training duration  $(timeSGD) seconds")
        
        p
    end
end

function testRPROP()
    dimensions = [1,4,1]
    net = RNN.MakeNetwork(dimensions)
    inputs = collect(0:0.1:1)'
    outputs = map(x->(sin(2pi*x)+1)/2, inputs)
    N = size(inputs)[2]
    unknowns = rand(Uniform(0,1),10*N)'
    errorsRPROP = RNN.RPROP(net, inputs, outputs, 0.1, 50.0, 50000)
    estimation = [RNN.forwardPropagation(net, unknowns[:,i])[end][2] for i in 1:10*N]'

    plot(layer(x=inputs, y=outputs, Geom.line, Theme(default_color = RGB(0,0,0))),
         layer(x=unknowns, y=estimation, Geom.point, Theme(default_color = RGB(1,0,0))),
         Guide.title("RPROP test"),
         Guide.xlabel("x"),
         Guide.ylabel("(sin(2πx)+1)/2"))
end
