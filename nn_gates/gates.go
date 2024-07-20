package main

import (
	"fmt"
	"math"
	"math/rand"
)

// Our matrix struct contains the number of rows, the number of columns, the stride (number of
// elements to skip to move to next row, usually equal to columns), and es (in C, a pointer to
// the first element, but in this just a handle to the slice of numbers.) Data is stored as a
// list of
type mat struct {
	rows   int
	cols   int
	stride int
	es     []float64
}

func (m *mat) At(i, j int) float64 {
	return m.es[i*m.stride+j]
}

var andGate = []float64{
	0, 0, 0,
	1, 0, 0,
	0, 1, 0,
	1, 1, 1,
}

var orGate = []float64{
	0, 0, 0,
	1, 0, 1,
	0, 1, 1,
	1, 1, 1,
}

var nandGate = []float64{
	0, 0, 1,
	1, 0, 1,
	0, 1, 1,
	1, 1, 0,
}

var xorGate = []float64{
	0, 0, 0,
	1, 0, 1,
	0, 1, 1,
	1, 1, 0,
}

var instances = [][]float64{
	andGate,
	orGate,
	nandGate,
	xorGate,
}

var train mat = mat{
	rows:   4,
	cols:   3,
	stride: 3,
	es:     andGate,
}

func newMat(rows, cols, stride int, data []float64) mat {
	var m mat = mat{
		rows:   4,
		cols:   3,
		stride: 3,
		es:     data,
	}
	return m
}

var train_count int = 4

// Calculates the cost of our two weights and bias, wanting as close as 0 as possible.
func cost(w1, w2, b float64, m mat) float64 {
	result := 0.0
	for i := 0; i < train_count; i++ {
		x1 := m.At(i, 0)
		x2 := m.At(i, 1)
		y := sigmoid(x1*w1 + x2*w2 + b)
		d := y - m.At(i, 2)
		result += d * d
	}
	return result / float64(train_count)
}

// Calculates the amount we should change our weights and biases using the finite different method.
func dcost(epsilon, w1, w2, b float64, m mat) (dw1, dw2, db float64) {

	c := cost(w1, w2, b, m)
	dw1 = (cost(w1+epsilon, w2, b, m) - c) / epsilon
	dw2 = (cost(w1, w2+epsilon, b, m) - c) / epsilon
	db = (cost(w1, w2, b+epsilon, m) - c) / epsilon

	return dw1, dw2, db
}

func gcost(w1, w2, b float64, m mat) (dw1, dw2, db float64) {
	dw1, dw2, db = 0, 0, 0
	for i := 0; i < m.rows; i++ {
		xi := m.At(i, 0)
		yi := m.At(i, 1)
		zi := m.At(i, 2)
		ai := sigmoid(w1*xi + w2*yi + b)
		di := 2 * (ai - zi) * ai * (1 - ai)

		dw1 += di * xi
		dw2 += di * yi
		db += di
	}
	dw1 /= float64(train_count)
	dw2 /= float64(train_count)
	db /= float64(train_count)

	return dw1, dw2, db
}

func randFloat() float64 {
	return rand.Float64()
}

func sigmoid(x float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-x))
}

func main() {

	for k := 0; k < len(instances); k++ {
		w1 := randFloat()
		w2 := randFloat()
		b := randFloat()
		mat := newMat(4, 3, 3, instances[k])
		//fmt.Printf("seed w1: %v, seed w2: %v, seed b: %v\n", w1, w2, b)
		fmt.Printf("initial cost: %v\n", cost(w1, w2, b, mat))

		for i := 1; i < 10*1000; i++ {
			dw1, dw2, db := gcost(w1, w2, b, mat)
			w1 -= dw1
			w2 -= dw2
			b -= db
			//fmt.Printf("Cost: %v\n", cost(w1, w2, b, mat))
		}
		fmt.Printf("end cost: %v\n", cost(w1, w2, b, mat))

		//fmt.Printf("w1 = %v, w2 = %v, b = %v\n", w1, w2, b)

		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				fmt.Printf("%v | %v = %v \n", i, j, sigmoid(float64(i)*w1+float64(j)*w2+b))
			}
		}

	}

}
