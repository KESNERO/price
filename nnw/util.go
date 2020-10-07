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

func Sigmoid(x float64) float64 {
	y := 1 / (1+math.Pow(math.E, -x))
	return y
}

func LeRU(x float64) float64 {
	y := math.Max(0, x)
	return y
}

func DeSigmoid(y float64) float64 {
	z := (1-y)/y
	x := -math.Log(z)
	return x
}

func DeLeRU(y float64) float64 {
	switch y > 0 {
	case true:
		return y
	default:
		return 0
	}
}

func SigmoidDerivative(y float64) float64 {
	return y * (1-y)
}

func LeRUDerivative(y float64) float64 {
	return 1
}

func Sum(num []float64) float64 {
	s := 0.0
	for _, n := range num {
		s += n
	}
	return s
}