package nnw

type Layer struct {
	size int
	neurons []*Neuron
	prev, next *Layer
}

func NewLayer(inSize, s int, at string) *Layer {
	var l = new(Layer)
	l.size = s
	l.prev = nil
	l.next = nil
	l.neurons = make([]*Neuron, l.size)
	for i := range l.neurons {
		l.neurons[i] = NewNeuron(inSize, at)
	}
	return l
}

func (l *Layer) Input(in []float64) {
	for _, n := range l.neurons {
		n.SetInput(in)
	}
}

func (l *Layer) Output() []float64 {
	out := make([]float64, l.size)
	for i := range l.neurons {
		out[i] = l.neurons[i].out
	}
	return out
}

func (l *Layer) ResetDelta() {
	for _, n := range l.neurons {
		for i := 0; i < n.size; i++ {
			n.delta[i] = 0
		}
	}
}

func (l *Layer) AverageDelta(batchSize int) {
	for _, n := range l.neurons {
		for i := 0; i < n.size; i++ {
			n.delta[i] /= float64(batchSize)
		}
	}
}

func (l *Layer) UpdateWeight(learningRate float64) {
	for k := range l.neurons {
		n := l.neurons[k]
		for i := 0; i < n.size; i++ {
			//fmt.Printf("w[%v][%v]: %f, delta[%v][%v]: %f, after: %f\n", i, k, n.weight[i], i, k, n.delta[i], n.delta[i]*learningRate)
			n.weight[i] -= n.delta[i] * learningRate
		}
	}
}

func (l *Layer) ErrorDerivative(t []float64) []float64 {
	ed := make([]float64, len(t))
	for i := range t {
		ed[i] = l.neurons[i].ErrorDerivative(t[i])
	}
	return ed
}

func (l *Layer) Error(t []float64) []float64 {
	err := make([]float64, len(t))
	for i, n := range l.neurons {
		err[i] = n.Error(t[i])
	}
	return err
}

func (l *Layer) Forward(in []float64) {
	l.Input(in)
	for _, n := range l.neurons {
		n.CalculateInput()
		n.Activate()
	}
	if l.next != nil {
		out := l.Output()
		l.next.Forward(out)
	}
}

func (l *Layer) BackPropagation(in []float64, errDerivative []float64) {
	if l.prev != nil {
		prev := l.prev
		prevErrDerivative := make([]float64, prev.size)
		for i, n := range l.neurons {
			for j := 0; j < n.size; j++ {
				n.delta[j] += errDerivative[i] * n.ActivationDerivative() * n.WeightDerivative(j)
				prevErrDerivative[j] += errDerivative[i] * n.ActivationDerivative() * n.weight[j]
			}
		}
		prev.BackPropagation(in, prevErrDerivative)
	} else {
		for i, n := range l.neurons {
			for j := 0; j < n.size; j++ {
				n.delta[j] += errDerivative[i] * n.ActivationDerivative() * n.WeightDerivative(j)
			}
		}
	}
}