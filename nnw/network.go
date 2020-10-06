package nnw

type Network struct {
	BatchSize     int
	LearningRate  float64
	Layers        []*Layer
	inputLayer *Layer
	outputLayer *Layer
}

func NewNetwork(inSize int, layerDefine []int, bs int, lr float64, at string) *Network {
	network := new(Network)
	network.BatchSize = bs
	network.LearningRate = lr
	for i, ln := range layerDefine {
		if i == 0 {
			l := NewLayer(inSize, ln, at)
			network.inputLayer = l
			network.Layers = append(network.Layers, l)
		} else {
			l := NewLayer(layerDefine[i-1], ln, at)
			network.Layers = append(network.Layers, l)
			l.prev = network.Layers[i-1]
			l.prev.next = l
		}
	}
	network.outputLayer = network.Layers[len(layerDefine)-1]
	return network
}

func (network *Network) Train(trainData, expected [][]float64, step int) {
	for t := 0; t < step; t++ {
		for i := 1; i < len(trainData); i++ {
			network.inputLayer.Forward(trainData[i])
			network.outputLayer.BackPropagation(trainData[i], network.outputLayer.ErrorDerivative(expected[i]))
			if (i+1) % network.BatchSize == 0 {
				for j := len(network.Layers)-1; j >= 0; j-- {
					network.Layers[j].AverageDelta(network.BatchSize)
					network.Layers[j].UpdateWeight(network.LearningRate)
					network.Layers[j].ResetDelta()
				}
			} else if i+1 == len(trainData) {
				for j := len(network.Layers)-1; j >= 0; j-- {
					network.Layers[j].AverageDelta(len(trainData) % network.BatchSize)
					network.Layers[j].UpdateWeight(network.LearningRate)
					network.Layers[j].ResetDelta()
				}
			}
		}
	}
}

func (network *Network) Predict(in [][]float64) [][]float64 {
	out := make([][]float64, 0)
	for i := range in {
		network.inputLayer.Forward(in[i])
		out = append(out, network.outputLayer.Output())
	}
	return out
}