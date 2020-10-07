package main

import (
	"encoding/csv"
	"fmt"
	"github.com/KESNERO/price/nnw"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func LoadData(filename string) [][]float64 {
	f, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	r := csv.NewReader(f)
	record, _ := r.ReadAll()
	allData := make([][]float64, len(record))
	for i, line := range record {
		allData[i] = make([]float64, len(line)+1)
		for j, ele := range line {
			if j == 0 {
				ts := strings.Split(ele, "-")
				allData[i][0], err = strconv.ParseFloat(ts[0], 64)
				if err != nil {
					panic(err)
				}
				allData[i][1], err = strconv.ParseFloat(ts[1], 64)
				if err != nil {
					panic(err)
				}
			} else {
				allData[i][j+1], err = strconv.ParseFloat(ele, 64)
				if err != nil {
					panic(err)
				}
			}
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

func PreprocessData(data [][]float64) (date [][]string, in, out [][]float64) {
	in = SplitColumn(data, 0, 6)
	out = SplitColumn(data, 6, 8)
	date = make([][]string, len(data))
	for i, row := range in {
		tmp := make([]float64, len(row))
		date[i] = make([]string, 2)
		date[i][0] = fmt.Sprintf("%v", row[0])
		date[i][1] = fmt.Sprintf("%v", row[1])
		tmp[0] = (row[0]-2012) / 10.0
		tmp[1] = row[1] / 12
		tmp[2] = row[2] / 10000.0
		tmp[3] = row[3] / 30000.0
		tmp[4] = row[4] / 50000.0
		tmp[5] = row[5] / 50000.0
		in[i] = tmp[2:4]
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
		row[0] = nnw.LeRU(row[0])
		row[1] = nnw.LeRU(row[1])
	}
}

func Compare(date [][]string, in, out, expected [][]float64, print bool) {
	if print {
		totalError := make([]float64, 2)
		for i := range out {
			date := fmt.Sprintf("%v-%02v", in[i][0]*10+2012, in[i][1]*12)
			totalError[0] += math.Pow(out[i][0]*50000.0 - expected[i][0]*50000.0, 2.0)
			fmt.Printf("%v low predict: %f, actual: %f\n", date, out[i][0]*50000.0, expected[i][1]*50000.0)
			totalError[1] += math.Pow(out[i][1]*50000.0 - expected[i][1]*50000.0, 2.0)
			fmt.Printf("%v ave predict: %f, actual: %f\n", date, out[i][1]*50000.0, expected[i][1]*50000.0)
		}
		totalError[0] /= float64(len(expected))
		totalError[1] /= float64(len(expected))
		totalError[0] = math.Sqrt(totalError[0])
		totalError[1] = math.Sqrt(totalError[1])
		fmt.Printf("low price variance: %f; average price variance: %f\n", totalError[0], totalError[1])
	}
}

func main() {
	allData := LoadData("data.csv")
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(allData), func(i, j int) {
		allData[i], allData[j] = allData[j], allData[i]
	})
	trainSize := len(allData)/2
	trainData := allData[0:trainSize]
	date, in, expected := PreprocessData(trainData)

	// Three key point parameters
	bs := 10
	step := 1000000
	learningRate := 0.0001

	inSize := len(in[0])
	outSize := len(expected[0])
	layerDefine := []int{inSize, 6, outSize}    // {inputSize, hiddenSize, outputSize}
	fn := "LeRU"

	network := nnw.NewNetwork(inSize, outSize, layerDefine, bs, learningRate, fn)
	network.Train(in, expected, step)

	out := network.Predict(in)
	Compare(date, in, out, expected, true)

	testData := allData[trainSize:]
	date, in, expected = PreprocessData(testData)
	out = network.Predict(in)
	Compare(date, in, out, expected, true)
}
