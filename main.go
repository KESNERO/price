package main

import (
	"encoding/csv"
	"fmt"
	"github.com/KESNERO/price/nnw"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func NormalizeData(allData [][]float64) [][]float64 {
	var result = make([][]float64, len(allData))
	for i := 0; i < len(allData); i++ {
		result[i] = make([]float64, len(allData[i]))
		result[i][0] = allData[i][0] / 10000.0        // lastAvailable
		result[i][1] = allData[i][1] / 30000.0        // lastCandidate
		result[i][2] = allData[i][2] / 50000.0        // lastLowPrice
		result[i][3] = allData[i][3] / 50000.0        // lastAveragePrice
		result[i][4] = allData[i][4] / 500.0          // lastLowPriceNumber
		result[i][5] = allData[i][5] / 500.0          // lastLowPriceSuccessNumber
		result[i][6] = (allData[i][6] - 2012.0) / 8.0 // year
		result[i][7] = allData[i][7] / 12.0           // month
		result[i][8] = allData[i][8] / 10000.0        // available
		result[i][9] = allData[i][9] / 30000.0        // candidate
		result[i][10] = allData[i][10] / 50000.0      // firstPrice
		result[i][11] = allData[i][11] / 50000.0      // secondPrice
		result[i][12] = allData[i][12] / 50000.0      // resultLowPrice
		result[i][13] = allData[i][13] / 50000.0      // resultAveragePrice
	}
	return result
}

func CalculateError(expected, predict [][]float64) [][]float64 {
	err := make([][]float64, len(expected))
	for i := range expected {
		err[i] = make([]float64, len(expected[i]))
		for j := range expected[i] {
			err[i][j] = predict[i][j] - expected[i][j]
		}
	}
	return err
}

func CalculateVariance(expected, predict [][]float64) []float64 {
	var result = []float64{0.0, 0.0}
	for i := 0; i < len(result); i++ {
		for k := 0; k < len(expected); k++ {
			result[i] += math.Abs(predict[k][i]-expected[k][i])
		}
		result[i] /= float64(len(expected))
	}
	return result
}

func MinInt(a, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}

func SplitColumn(data [][]float64, from, to int) [][]float64 {
	result := make([][]float64, len(data))
	for i := range data {
		result[i] = make([]float64, to-from)
		copy(result[i], data[i][from:to])
	}
	return result
}

func RecoverData(normData [][]float64) [][]float64 {
	var result = make([][]float64, len(normData))
	for i := range normData {
		result[i] = make([]float64, len(normData[i]))
		result[i][0] = normData[i][0] * 50000.0 // lowPrice
		result[i][1] = normData[i][1] * 50000.0 // averagePrice
	}
	return result
}

func TestNetwork(dataSet [][]float64, network *nnw.Network, printPredict bool) {
	in := SplitColumn(dataSet, 0, 12)
	out := SplitColumn(dataSet, 12, 14)
	out = RecoverData(out)
	predict := network.Predict(in)
	predict = RecoverData(predict)
	if printPredict {
		for i := range predict {
			fmt.Printf("data %v-%v: [low predict: %f, real: %f]; [ave predict: %f, real %f]\n", int(in[i][6]*8)+2012, int(in[i][7]*12), predict[i][0], out[i][0], predict[i][1], out[i][1])
		}
	}
	err := CalculateVariance(out, predict)
	fmt.Printf("low: %f, ave: %f\n", err[0], err[1])
}

func LoadData(filename string) [][]float64 {
	f, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	r := csv.NewReader(f)
	allData := make([][]float64, 0)
	var lastRecord = make([]string, 0)
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			panic(err)
		}
		if len(lastRecord) == 0 {
			lastRecord = make([]string, 14)
			copy(lastRecord, record)
			continue
		}
		if len(record) == 0 {
			continue
		}
		splitString := strings.Split(record[0], "-")
		yearString := splitString[0]
		monthString := splitString[1]
		row := make([]float64, 0)
		lastAvailable, _ := strconv.ParseFloat(lastRecord[1], 64)
		row = append(row, lastAvailable)
		lastCandidates, _ := strconv.ParseFloat(lastRecord[2], 64)
		row = append(row, lastCandidates)
		lastLowPrice, _ := strconv.ParseFloat(lastRecord[5], 64)
		row = append(row, lastLowPrice)
		lastAveragePrice, _ := strconv.ParseFloat(lastRecord[6], 64)
		row = append(row, lastAveragePrice)
		lastLowNum, _ := strconv.ParseFloat(lastRecord[7], 64)
		row = append(row, lastLowNum)
		lastLowSuccessNum, _ := strconv.ParseFloat(lastRecord[8], 64)
		row = append(row, lastLowSuccessNum)
		year, _ := strconv.ParseFloat(yearString, 64)
		row = append(row, year)
		month, _ := strconv.ParseFloat(monthString, 64)
		row = append(row, month)
		curAvailable, _ := strconv.ParseFloat(record[1], 64)
		row = append(row, curAvailable)
		curCandidates, _ := strconv.ParseFloat(record[2], 64)
		row = append(row, curCandidates)
		firstPrice, _ := strconv.ParseFloat(record[3], 64)
		row = append(row, firstPrice)
		secondPrice, _ := strconv.ParseFloat(record[4], 64)
		row = append(row, secondPrice)
		lowPrice, _ := strconv.ParseFloat(record[5], 64)
		row = append(row, lowPrice)
		averagePrice, _ := strconv.ParseFloat(record[6], 64)
		row = append(row, averagePrice)
		allData = append(allData, row)
		copy(lastRecord, record)
	}
	return allData
}

func main() {
	bs := 1
	step := 100000
	inSize := 12
	outSize := 2
	learningRate := 0.00001
	allData := LoadData("data.csv")
	allData = NormalizeData(allData)
	trainSize := len(allData)/3*2
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(allData), func(i, j int) {
		allData[i], allData[j] = allData[j], allData[i]
	})
	network := nnw.NewNetwork([]int{inSize, 24, 12, 6, outSize}, bs, learningRate)
	trainData := allData[0:trainSize]
	testData := allData[trainSize:]
	for k := 0; k < step; k++ {
		rand.Shuffle(len(trainData), func(i, j int) {
			trainData[i], trainData[j] = trainData[j], trainData[i]
		})
		for i := 0; i < len(trainData); i += bs {
			rightBound := MinInt(i+bs, len(trainData))
			data := trainData[i:rightBound]
			in := SplitColumn(data, 0, inSize)
			out := SplitColumn(data, inSize, inSize+outSize)
			predict := network.ForwardSpread(in)
			err := CalculateError(out, predict)
			network.BackPropagation(err)
		}
		step--
	}
	fmt.Println("Error on trainData: ")
	TestNetwork(trainData, network, false)
	fmt.Println("Error on testData: ")
	TestNetwork(testData, network, true)
}
