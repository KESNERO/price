package nnw

import "fmt"

type Network struct {
	BatchSize     int
	InSize, OutSize int
	LearningRate  float64
	FunctionName string
	Layers        []*Layer
	InputLayer, OutputLayer *Layer
}

func NewNetwork(inSize, outSize int, layerDefine []int, bs int, lr float64, fn string) *Network {
	network := new(Network)
	network.InSize = inSize
	network.OutSize = outSize
	network.BatchSize = bs
	network.LearningRate = lr
	for i, size := range layerDefine {
		if i == 0 {
			l := NewLayer(inSize, size, fn, i)
			network.InputLayer = l
			network.Layers = append(network.Layers, l)
		} else {
			l := NewLayer(layerDefine[i-1], size, fn, i)
			l.prev = network.Layers[i-1]
			l.prev.next = l
			network.Layers = append(network.Layers, l)
		}
	}
	network.OutputLayer = network.Layers[len(network.Layers)-1]
	return network
}

func (network *Network) Train(trainInput, expected [][]float64, step int) {
	for t := 0; t < step; t++ {
		for _, l := range network.Layers {
			l.ResetDelta()
		}
		network.OutputLayer.ResetDelta()
		for i := 0; i < len(trainInput); i++ {
			network.InputLayer.Forward(trainInput[i])
			output := network.OutputLayer.Output()
			bias := make([]float64, len(output))
			for j := range bias {
				bias[j] = output[j] - expected[i][j]
			}
			network.OutputLayer.BackPropagation(bias)
			if (i+1) % network.BatchSize == 0 {
				for _, l := range network.Layers {
					l.AverageDelta(network.BatchSize)
					l.UpdateWeight(network.LearningRate)
					l.ResetDelta()
				}
				network.OutputLayer.AverageDelta(network.BatchSize)
				network.OutputLayer.UpdateWeight(network.LearningRate)
				network.OutputLayer.ResetDelta()
			}
		}
	}
}

func (network *Network) Predict(in [][]float64) [][]float64 {
	out := make([][]float64, 0)
	for i := range in {
		network.InputLayer.Forward(in[i])
		out = append(out, network.OutputLayer.Output())
	}
	fmt.Println(out)
	return out
}