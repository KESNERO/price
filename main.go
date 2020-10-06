package main

import (
	"encoding/csv"
	"fmt"
	"github.com/KESNERO/price/nnw"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func Sigmoid(x float64) float64 {
	y := 1 / (1+math.Pow(math.E, -x))
	return y
}

func LoadData(filename string) [][]float64 {
	f, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	r := csv.NewReader(f)
	record, _ := r.ReadAll()
	allData := make([][]float64, len(record))
	for i, line := range record {
		allData[i] = make([]float64, len(line))
		for j, ele := range line {
			allData[i][j], _ = strconv.ParseFloat(ele, 64)
		}
	}
	return allData
}

func SplitColumn(data [][]float64, from, to int) [][]float64 {
	result := make([][]float64, len(data))
	for i := range data {
		result[i] = make([]float64, to-from)
		copy(result[i], data[i][from:to])
	}
	return result
}

func PreprocessData(data [][]float64) (in, out [][]float64) {
	in = SplitColumn(data, 1, 5)
	out = SplitColumn(data, 5, 7)
	for _, row := range in {
		row[0] /= 10000.0
		row[1] /= 30000.0
		row[2] /= 50000.0
		row[3] /= 50000.0
	}
	for _, row := range out {
		row[0] /= 50000.0
		row[1] /= 50000.0
	}
	PreprocessOutput(out)
	return
}

func PreprocessOutput(data [][]float64) {
	for _, row := range data {
		row[0] = Sigmoid(row[0])
		row[1] = Sigmoid(row[1])
	}
}

func Compare(out, expected [][]float64, print bool) {
	if print {
		for i := range out {
			fmt.Printf("low predict: %f, actual: %f\n", out[i][0]*50000.0, expected[i][0]*50000.0)
			fmt.Printf("ave predict: %f, actual: %f\n", out[i][1]*50000.0, expected[i][1]*50000.0)
		}
	}
}

func main() {
	bs := 1
	step := 100000
	inSize := 4
	outSize := 2
	learningRate := 0.0001
	printResult := true
	at := "LeRU"
	allData := LoadData("data.csv")
	trainSize := len(allData)/2
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(allData), func(i, j int) {
		allData[i], allData[j] = allData[j], allData[i]
	})
	network := nnw.NewNetwork(inSize, []int{4, outSize}, bs, learningRate, at)
	trainData := allData[0:trainSize]
	testData := allData[trainSize:]
	in, expected := PreprocessData(trainData)
	network.Train(in, expected, step)
	out := network.Predict(in)
	Compare(out, expected, printResult)
	in, expected = PreprocessData(testData)
	out = network.Predict(in)
	Compare(out, expected, printResult)
}
