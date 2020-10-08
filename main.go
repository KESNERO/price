package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"github.com/KESNERO/price/nnw"
	"io"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func Run(c *exec.Cmd) string {
	stdout, _ := c.StdoutPipe()
	stderr, _ := c.StderrPipe()
	defer stdout.Close()
	defer stderr.Close()
	if err := c.Start(); err != nil {
		panic(err)
	}
	buf := new(bytes.Buffer)
	buf.ReadFrom(stdout)
	out := buf.String()
	io.Copy(os.Stderr, stderr)
	return out
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

func GetX(y float64) float64 {
	bin, _ := exec.LookPath("python3")
	cmd := exec.Command(bin, "calx.py", fmt.Sprintf("%f", y))
	result := Run(cmd)
	result = strings.Split(result, "\n")[0]
	x, err := strconv.ParseFloat(result, 64)
	if err != nil {
		panic(err)
	}
	return x
}

func CalculateAllAverage(available, candidate, low, ave float64) (mu, sigma float64) {
	if available >= candidate {
		mu = ave
		sigma = ave - low
		return
	}
	lowP := available / candidate
	aveP := lowP / 2
	y1 := 1 - lowP
	y2 := 1 - lowP + aveP
	x1 := GetX(y1)
	x2 := GetX(y2)
	sigma = (ave-low) / (x2-x1)
	mu = low - x1 * sigma
	return
}

func PreprocessData1(data [][]float64) (date [][]string, in, out [][]float64) {
	date = make([][]string, len(data))
	in = make([][]float64, len(data))
	out = make([][]float64, len(data))
	for i, row := range data {
		date[i] = make([]string, 2)
		in[i] = make([]float64, 5)
		out[i] = make([]float64, 2)
		date[i][0] = fmt.Sprintf("%v", row[0])
		date[i][1] = fmt.Sprintf("%v", row[1])
		in[i][0] = row[2] / 10000
		in[i][1] = row[3] / 50000
		in[i][2] = row[2] / row[3]
		in[i][3] = row[4] / 50000
		in[i][4] = row[5] / 50000
		ave, bias := CalculateAllAverage(row[2], row[3], row[6], row[7])
		out[i][0], out[i][1] = ave / 50000, bias / 50000
	}
	return
}

func PreprocessData2(data [][]float64) (date [][]string, in, out [][]float64) {
	date = make([][]string, len(data))
	in = make([][]float64, len(data))
	out = make([][]float64, len(data))
	for i, row := range data {
		date[i] = make([]string, 2)
		in[i] = make([]float64, 5)
		out[i] = make([]float64, 2)
		date[i][0] = fmt.Sprintf("%v", row[0])
		date[i][1] = fmt.Sprintf("%v", row[1])
		in[i][0] = row[2] / 10000
		in[i][1] = row[3] / 50000
		in[i][2] = row[2] / row[3]
		in[i][3] = row[4] / 50000
		in[i][4] = row[5] / 50000
		out[i][0], out[i][1] = row[6] / 50000, (row[6]-row[7]) / 50000
	}
	return
}

func Combind(in, preOut [][]float64) [][]float64 {
    result := make([][]float64, len(in))
    for i := range in {
        result[i] = make([]float64, 7)
        copy(result[i], in[i])
        result[i][5], result[i][6] = preOut[i][0], preOut[i][1]
    }
	return result
}

func RecoverOutput(out [][]float64) [][]float64{
	result := make([][]float64, len(out))
	for i := range out {
		result[i] = make([]float64, len(out[i]))
		result[i][0] = out[i][0] * 50000
		result[i][1] = out[i][1] * 50000
	}
	return result
}

func Variance(date [][]string, out, expected [][]float64) {
	size := len(out)
	v := make([]float64, len(out[0]))
	for i := 0; i < len(out); i++ {
		fmt.Printf("%v-%v: average predict %f, actual %f\n", date[i][0], date[i][1], out[i][0], expected[i][0])
		fmt.Printf("%v-%v: bias predict %f, actual %f\n", date[i][0], date[i][1], out[i][1], expected[i][1])
		v[0] += math.Pow(expected[i][0]-out[i][0], 2)
		v[1] += math.Pow(expected[i][1]-out[i][1], 2)
	}
	fmt.Printf("average variance: %f, bias variance: %f\n", math.Sqrt(v[0])/float64(size), math.Sqrt(v[1])/float64(size))
}

func main() {
	allData := LoadData("data.csv")
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(allData), func(i, j int) {
		allData[i], allData[j] = allData[j], allData[i]
	})
	trainSize := len(allData)/2

	// Four key point parameters
	bs := 1
	step := 100000
	learningRate := 0.001
	fn := "LeRU"

	trainData := allData[0:trainSize]
	date1, in1, expected1 := PreprocessData1(trainData)
	layerDefine1 := []int{len(in1[0]), 10, len(expected1[0])}    // {inputSize, hiddenSize, outputSize}

	network1 := nnw.NewNetwork(len(in1[0]), len(expected1[0]), layerDefine1, bs, learningRate, fn)
	network1.Train(in1, expected1, step)

	out1 := network1.Predict(in1)
	out1 = RecoverOutput(out1)
	expected1 = RecoverOutput(expected1)

	Variance(date1, out1, expected1)
}
