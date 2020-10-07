package nnw

import (
	"fmt"
)

type Layer struct {
	inSize, outSize, index int
	neurons []*Neuron
	fn string
	prev, next *Layer
}

func NewLayer(inSize, outSize int, fn string, layerIndex int) *Layer {
	var l = new(Layer)
	l.inSize = inSize
	l.outSize = outSize
	l.index = layerIndex
	l.fn = fn
	l.prev = nil
	l.next = nil
	l.neurons = make([]*Neuron, l.outSize)
	for i := range l.neurons {
		l.neurons[i] = NewNeuron(l.inSize)
	}
	return l
}

func (l *Layer) Input(in []float64) {
	for _, n := range l.neurons {
		n.SetInput(in)
	}
}


func (l *Layer) Calculate() {
	for _, n := range l.neurons {
		n.Calculate()
	}
}

func (l *Layer) Activate() {
	for _, n := range l.neurons {
		n.Activate(l.fn)
	}
}

func (l *Layer) Output() []float64 {
	out := make([]float64, l.outSize)
	for i := range l.neurons {
		out[i] = l.neurons[i].netOut
	}
	return out
}

func (l *Layer) ResetDelta() {
	for _, n := range l.neurons {
		n.ResetDelta()
	}
}

func (l *Layer) AverageDelta(batchSize int) {
	for _, n := range l.neurons {
		n.DivideDelta(float64(batchSize))
	}
}

func (l *Layer) UpdateWeight(learningRate float64) {
	for _, n := range l.neurons {
		n.UpdateWeight(learningRate)
	}
}

func (l *Layer) Forward(in []float64) {
	if len(in) != l.inSize {
		err := fmt.Errorf("input size didn't match layer defined insize")
		panic(err)
	}
	l.Input(in)
	l.Calculate()
	l.Activate()
	out := l.Output()
	if l.next != nil {
		l.next.Forward(out)
	}
}

func (l *Layer) BackPropagation(bias []float64) {
	if len(bias) != l.outSize {
		err := fmt.Errorf("err size didn't match layer defined outsize")
		panic(err)
	}
	nextBias := make([]float64, l.inSize)
	for j := range nextBias {
		for i := range l.neurons {
			nextBias[j] += bias[i] * (l.neurons[i].weight[j] / Sum(l.neurons[i].weight))
		}
		nextBias[j] /= float64(l.outSize)
	}
	for i := range l.neurons {
		l.neurons[i].UpdateDelta(l.fn, bias[i])
	}
	if l.prev != nil {
		l.prev.BackPropagation(nextBias)
	}
}

func (l *Layer) PrintOut() {
	fmt.Printf("layer %v's output:\n", l.index)
	for i, n := range l.neurons {
		fmt.Printf("neuron[%v]: %F\n", i, n.netOut)
	}
}

func (l *Layer) PrintIn() {
	fmt.Printf("layer %v's input:\n", l.index)
	for i, v := range l.neurons[0].in {
		fmt.Printf("in[%v]: %F\n", i, v)
	}
}

func (l *Layer) PrintWeight() {
	fmt.Printf("layer %v's weight:\n", l.index)
	for i, n := range l.neurons {
		for j := range n.weight {
			fmt.Printf("weight[%v][%v]: %F\n", j, i, n.weight[j])
		}
	}
}

func (l *Layer) PrintDelta() {
	fmt.Printf("layer %v's delta:\n", l.index)
	for i, n := range l.neurons {
		for j := range n.delta {
			fmt.Printf("delta[%v][%v]: %F\n", j, i, n.delta[j])
		}
	}
}