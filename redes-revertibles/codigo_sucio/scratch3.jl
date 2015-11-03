#############################
# NEURAL NETWORK PROCEDURES #
#############################

@doc """
Compute the logistic sigmoid function.
""" ->
function logistic(t)
    1.0./(1 .+ exp(-t))
end

@doc """
Compute the logistic function inverse, aka logit function.
""" ->
function logit(p)
    -log(1./p .- 1)
end

f = logistic
g = logit

@doc """
Forward propagation for one neuron.
""" ->
function sendInput(xs, ws)
    f(xs*ws)[1]
end

@doc """
Backward propagation for one neuron.
""" ->
function sendOutput(y, ws)
    sumResult = g(y)
    wBias     = ws[1]
    wRest     = ws[2:end]

    constantVal = sumResult - wBias

    xs = zeros(1,size(ws)[1])
    xs[1] = 1.0

    negativeTotal = 0
    possitiveTotal = 0
    for w in wRest
        if w < 0.0
            negativeTotal += w
        else
            possitiveTotal += w
        end
    end
    
    totalSum = 0.0
    for i in eachindex(ws)[2:end]
        w = ws[i]
        
        if w < 0
            negativeTotal-=w
        else
            possitiveTotal-=w
        end
        
        firstBound = (constantVal - negativeTotal - totalSum)/w
        secondBound = (constantVal - possitiveTotal - totalSum)/w
        
        low, high = cutInterval(firstBound, secondBound)
        
        x = rand()*(high-low) + low
        
        xs[i] = x
        totalSum += x*w
    end

    xs
end

########################
# AUXILIARY PROCEDURES #
########################

function cutInterval(a, b)
    lowBound  = min(a, b)
    highBound = max(a, b)
    (lowBound>0.0?lowBound:0.0 , highBound<1.0?highBound:1.0)
end

###########
# EXAMPLE #
###########

x0 = 1.0
x1 = 0.5
x2 = 1/3.0
x3 = 0.5
xs = [x0 x1 x2 x3]              # row vector

w0 = -1.0
w1 = 4.0
w2 = -6.0
w3 = 2.0
ws = [w0,w1,w2,w3]              # column vector

y = sendInput(xs, ws)
