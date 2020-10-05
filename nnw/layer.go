package nnw

import (
	"fmt"
	"math"
)

type Layer struct {
	Size   int
	Type   string
	Neural []float64
}

func Sigmoid(x float64) float64 {
	fmt.Printf("before sigmoid: %v", x)
	y := 1.0 / (1.0 + math.Exp(-x))
	fmt.Printf(", after sigmoid: %v\n", y)
	return y
}

func SigmoidDerivative(y float64) float64 {
	return y * (1 - y)
}

func LeRU(x float64) float64 {
	return math.Max(0, x)
}

func LeRUDerivative(y float64) float64 {
	if y >= 0 {
		return 1
	} else {
		return 0
	}
}

func NewLayer(s int, t string) *Layer {
	var layer = new(Layer)
	layer.Size = s
	layer.Type = t
	layer.Neural = make([]float64, layer.Size)
	return layer
}

func (l *Layer) LeftProduct(w [][]float64) []float64 {
	result := make([]float64, len(w))
	for i := range w {
		for j := range w[i] {
			result[i] += w[i][j] * l.Neural[j]
		}
	}
	return result
}

func (l *Layer) RightProduct(w [][]float64) []float64 {
	var result = make([]float64, len(w[0]))
	for i := range w {
		for j := range w[i] {
			result[j] += l.Neural[i] * w[i][j]
		}
	}
	return result
}

func (l *Layer) Activate() []float64 {
	var result = make([]float64, l.Size)
	switch l.Type {
	case "Sigmoid":
		for i := range l.Neural {
			result[i] = Sigmoid(l.Neural[i])
		}
	case "ReLU":
		for i := range l.Neural {
			result[i] = LeRU(l.Neural[i])
		}
	default:
		copy(result, l.Neural)
	}
	return result
}

func (l *Layer) DeActivate(funcName string) []float64 {
	var result = make([]float64, l.Size)
	switch funcName {
	case "Sigmoid":
		for i := range l.Neural {
			result[i] = SigmoidDerivative(l.Neural[i])
		}
	case "ReLU":
		for i := range l.Neural {
			result[i] = LeRUDerivative(l.Neural[i])
		}
	default:
		copy(result, l.Neural)
	}
	return result
}

func (l *Layer) Input(in []float64) {
	copy(l.Neural, in)
}

func (l *Layer) Output() []float64 {
	out := make([]float64, l.Size)
	copy(out, l.Neural)
	return out
}

func (l *Layer) Plus(in []float64) {
	for i := range l.Neural {
		l.Neural[i] += in[i]
	}
}

func (l *Layer) Divide(denominator float64) {
	for i := range l.Neural {
		l.Neural[i] /= denominator
	}
}

func (l *Layer) Reset() {
	for i := range l.Neural {
		l.Neural[i] = 0
	}
}

func (l *Layer) Variance(t []float64) []float64 {
	result := make([]float64, l.Size)
	for i := range l.Neural {
		result[i] = math.Pow(l.Neural[i]-t[i], 2.0)
	}
	return result
}

func (l *Layer) Print(batchIndex, curLayerIndex int) {
	fmt.Printf("Layer %v: ", curLayerIndex)
	for i := range l.Neural {
		if i > 0 {
			fmt.Printf(", ")
		}
		fmt.Printf("%v", l.Neural[i])
	}
	fmt.Println()
}