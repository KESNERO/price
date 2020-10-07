package nnw

type Neuron struct {
	inSize int
	in, weight, delta, out []float64
	outTotal, netOut float64
}

func NewNeuron(inSize int) *Neuron {
	n := new(Neuron)
	n.inSize = inSize
	n.in = make([]float64, n.inSize)
	n.weight = make([]float64, n.inSize)
	n.delta = make([]float64, n.inSize)
	n.out = make([]float64, n.inSize)
	for i := range n.weight {
		n.weight[i] = Random()
	}
	return n
}

func (n *Neuron) SetInput(in []float64) {
	copy(n.in, in)
}

func (n *Neuron) Calculate() {
	n.outTotal = 0.0
	for i := range n.in {
		n.out[i] = n.in[i] * n.weight[i]
		n.outTotal += n.out[i]
	}
}

func (n *Neuron) Activate(fn string) {
	switch fn {
	case "Sigmoid":
		n.netOut = Sigmoid(n.outTotal)
	case "LeRU":
		n.netOut = LeRU(n.outTotal)
	case "Normal":
		n.netOut = n.outTotal
	}
}

func (n *Neuron) ResetDelta() {
	for i := range n.delta {
		n.delta[i] = 0.0
	}
}

func (n *Neuron) DivideDelta(denominator float64) {
	for i := range n.delta {
		n.delta[i] /= denominator
	}
}

func (n *Neuron) BiasBack(fn string, bias float64) float64 {
	switch fn {
	case "Sigmoid":
		return DeSigmoid(bias)
	case "LeRU":
		return DeLeRU(bias)
	default:
		return bias
	}
}

func (n *Neuron) UpdateDelta(fn string, bias float64) {
	switch fn {
	case "Sigmoid":
		for i := range n.delta {
			n.delta[i] += bias * SigmoidDerivative(n.netOut) * n.in[i]
		}
	case "LeRU":
		for i := range n.delta {
			n.delta[i] += bias * LeRUDerivative(n.netOut) * n.in[i]
		}
	case "Normal":
		for i := range n.delta {
			n.delta[i] += bias * n.in[i]
		}
	}
}

func (n *Neuron) UpdateWeight(learningRate float64) {
	for i := range n.weight {
		n.weight[i] -= learningRate * n.delta[i]
	}
}