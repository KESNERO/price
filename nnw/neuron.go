package nnw

import (
	"math"
	"math/rand"
)

func Gauss() float64 {
	μ := 0.0
	σ := 0.25
	x := rand.Float64()
	w := 1 / (math.Sqrt(2*math.Pi) * σ) * math.Pow(math.E, -math.Pow(x-μ, 2)/(2*math.Pow(σ, 2)))
	return w
}

func Random() float64 {
	w := rand.Float64()
	return w
}

type Neuron struct {
	out float64
	size int
	in []float64
	weight []float64
	delta []float64
	activeType string
}

func NewNeuron(inSize int, at string) *Neuron {
	n := new(Neuron)
	n.size = inSize
	n.in = make([]float64, inSize)
	n.weight = make([]float64, inSize)
	n.delta = make([]float64, inSize)
	n.activeType = at
	for i := range n.weight {
		n.weight[i] = Gauss()
	}
	return n
}

func (n *Neuron) SetInput(in []float64) {
	copy(n.in, in)
}

func (n *Neuron) CalculateInput() {
	for _, in := range n.in {
		n.out += in
	}
}

func (n *Neuron) Activate() {
	switch n.activeType {
	case "Sigmoid":
		x := n.out
		y := 1 / (1+math.Pow(math.E, -x))
		n.out = y
	case "LeRU":
		x := n.out
		y := math.Max(0.0, x)
		n.out = y
	}
}

// d(error)/d(output)
func (n *Neuron) ErrorDerivative(t float64) float64 {
	return n.out - t
}

// d(output)/d(input)
func (n *Neuron) ActivationDerivative() float64 {
	switch n.activeType {
	case "Sigmoid":
		return n.out * (1 - n.out)
	case "LeRU":
		return 1.0
	default:
		return 1.0
	}
}

// d(input)/d(w)
func (n *Neuron) WeightDerivative(i int) float64 {
	return n.in[i]
}

func (n *Neuron) Error(t float64) float64 {
	return math.Pow(t-n.out, 2.0) / 2
}