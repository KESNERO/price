package nnw

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
)

func Gauss() float64 {
	μ := 0.0
	σ := 0.25
	x := rand.Float64()
	result := 1 / (math.Sqrt(2*math.Pi) * σ) * math.Pow(math.E, -math.Pow(x-μ, 2)/(2*math.Pow(σ, 2)))
	return result
}

type Network struct {
	LayerDefine   []int
	BatchSize     int
	LearningRate  float64
	Layers        [][]*Layer
	W             [][][]float64 // W[0][1][2] means from first layer's second position
	// to next layer's third position
}

func NewNetwork(ld []int, bs int, lr float64) *Network {
	network := new(Network)
	network.LayerDefine = ld
	network.BatchSize = bs
	network.LearningRate = lr
	network.Layers = make([][]*Layer, network.BatchSize)
	network.W = make([][][]float64, 0)
	for k := 0; k < network.BatchSize; k++ {
		network.Layers[k] = make([]*Layer, len(ld))
		for j := range network.Layers[k] {
			switch j {
			case 0:
				network.Layers[k][j] = NewLayer(ld[j], "Normal")
			case len(ld)-1:
				network.Layers[k][j] = NewLayer(ld[j], "LeRU")
			default:
				network.Layers[k][j] = NewLayer(ld[j], "LeRU")
			}
		}
	}
	for i, ln := range ld {
		if i > 0 {
			w := make([][]float64, ld[i-1])
			for j := range w {
				w[j] = make([]float64, ln)
				for k := range w[j] {
					w[j][k] = Gauss()
				}
			}
			network.W = append(network.W, w)
		}
	}
	return network
}

func (network *Network) ForwardSpread(input [][]float64) [][]float64 {
	var output = make([][]float64, len(input))
	for i := range input {
		curLayerIndex := 0
		curWIndex := 0
		network.Layers[i][curLayerIndex].Input(input[i])
		for curLayerIndex < len(network.LayerDefine)-1 {
			//network.Layers[i][curLayerIndex].Print(i, curLayerIndex)
			network.Layers[i][curLayerIndex+1].Input(network.Layers[i][curLayerIndex].RightProduct(network.W[curWIndex]))
			network.Layers[i][curLayerIndex+1].Activate()
			curLayerIndex++
			curWIndex++
		}
		//network.Layers[i][curLayerIndex].Print(i, curLayerIndex)
		output[i] = make([]float64, network.Layers[i][curLayerIndex].Size)
		copy(output[i], network.Layers[i][curLayerIndex].Output())
	}
	return output
}

func (network *Network) UpdateW(err [][]float64, wIndex, leftIndex, rightIndex int) {
	tmp := make([][]float64, len(network.W[wIndex]))
	for i := range tmp {
		tmp[i] = make([]float64, len(network.W[wIndex][i]))
	}
	var leftSize, rightSize int
	for k := 0; k < len(err); k++ {
		leftSize = network.Layers[k][leftIndex].Size
		rightSize = network.Layers[k][rightIndex].Size
		for i := 0; i < leftSize; i++ {
			for j := 0; j < rightSize; j++ {
				y := network.Layers[k][rightIndex].Neural[j]
				x := network.Layers[k][leftIndex].Neural[i]
				e := err[k][j]
				switch network.Layers[k][leftIndex].Type {
				case "Sigmoid":
					tmp[i][j] += e * (1-y) * y * x
				default:
					tmp[i][j] += e * x
				}
			}
		}
	}
	for i := 0; i < leftSize; i++ {
		for j := 0; j < rightSize; j++ {
			delta := tmp[i][j] / float64(network.BatchSize)
			network.W[wIndex][i][j] -= network.LearningRate * network.W[wIndex][i][j] * delta
		}
	}
}

func (network *Network) NextError(err [][]float64, wIndex, leftIndex, rightIndex int) [][]float64 {
	nextError := make([][]float64, network.BatchSize)
	for k := range nextError {
		nextError[k] = make([]float64, network.Layers[k][leftIndex].Size)
	}
	var leftSize, rightSize int
	for k := 0; k < len(err); k++ {
		leftSize = network.Layers[k][leftIndex].Size
		rightSize = network.Layers[k][rightIndex].Size
		for i := 0; i < leftSize; i++ {
			for j := 0; j < rightSize; j++ {
				y := network.Layers[k][rightIndex].Neural[j]
				e := err[k][j]
				switch network.Layers[k][leftIndex].Type {
				case "Sigmoid":
					nextError[k][i] += e * y * (1-y) * network.W[wIndex][i][j]
				default:
					nextError[k][i] += e * network.W[wIndex][i][j]
				}
			}
		}
	}
	return nextError
}

func (network *Network) BackPropagation(err [][]float64) {
	curLayerIndex := len(network.LayerDefine) - 1
	curWIndex := len(network.W) - 1
	curError := make([][]float64, len(err))
	copy(curError, err)
	for curLayerIndex > 0 {
		nextErr := network.NextError(curError, curWIndex, curLayerIndex-1, curLayerIndex)
		network.UpdateW(curError, curWIndex, curLayerIndex-1, curLayerIndex)
		curError = make([][]float64, len(nextErr))
		copy(curError, nextErr)
		curLayerIndex--
		curWIndex--
	}
}

func (network *Network) Predict(in [][]float64) [][]float64 {
	out := make([][]float64, len(in))
	for i := range in {
		curIn := in[i:i+1]
		curOut := network.ForwardSpread(curIn)
		out[i] = make([]float64, len(curOut[0]))
		copy(out[i], curOut[0])
	}
	return out
}

func (network *Network) SaveW() {
	f, err := os.OpenFile("w.csv", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		panic(err)
	}
	stream, err := json.Marshal(network.W)
	if err != nil {
		panic(err)
	}
	_, _ = io.Copy(f, bytes.NewReader(stream))
}

func (network *Network) LoadW() {
	f, err := os.OpenFile("w.csv", os.O_RDONLY, 0644)
	if err != nil {
		panic(err)
	}
	stream, err := ioutil.ReadAll(f)
	if err != nil {
		panic(err)
	}
	if err = json.Unmarshal(stream, network.W); err != nil {
		panic(err)
	}
}
