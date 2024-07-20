package main

import (
	"fmt"
	"liz/assert"
	"liz/utils"
	"math"
	"math/rand"
)

type mat struct {
	rows   int
	cols   int
	stride int
	es     []float64
}

type nn struct {
	count       int
	activations []mat //activations will be count+1 because we must load the initial values.
	weights     []mat
	biases      []mat
}

func (m *mat) At(row, col int) *float64 {
	return &m.es[row*m.stride+col]
}

func newMat(r, c, s int, e []float64) mat {
	var m mat = mat{
		rows:   r,
		cols:   c,
		stride: s,
		es:     e,
	}
	return m
}

// Calculates the cost of our two weights and bias, wanting as close as 0 as possible.
func cost(w1, w2, b float64, m *mat) float64 {
	result := 0.0
	for i := 0; i < m.rows; i++ {
		x1 := *m.At(i, 0)
		x2 := *m.At(i, 1)
		y := sigmoid(x1*w1 + x2*w2 + b)
		d := y - *m.At(i, 2)
		result += d * d
	}
	return result / float64(m.rows)
}

func gcost(w1, w2, b float64, m *mat) (dw1, dw2, db float64) {
	dw1, dw2, db = 0, 0, 0
	for i := 0; i < m.rows; i++ {
		xi := *m.At(i, 0)
		yi := *m.At(i, 1)
		zi := *m.At(i, 2)
		ai := sigmoid(w1*xi + w2*yi + b)
		di := 2 * (ai - zi) * ai * (1 - ai)

		dw1 += di * xi
		dw2 += di * yi
		db += di
	}
	dw1 /= float64(m.rows)
	dw2 /= float64(m.rows)
	db /= float64(m.rows)

	return dw1, dw2, db
}

func randFloat() float64 {
	return rand.Float64()
}

func sigmoid(x float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-x))
}

func leakyReLU(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0.01 * x
	}
}

func derivLeakyReLU(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0.01
	}
}

func matRow(m *mat, row int) *mat {
	start := row * m.stride
	return &mat{
		rows:   1,
		cols:   m.cols,
		stride: m.stride,
		es:     m.es[start : start+m.cols],
	}
}

func matCol(m *mat, col int) mat {
	matrix := mat{
		rows:   m.rows,
		cols:   1,
		stride: 1,
		es:     make([]float64, m.rows),
	}
	for i := 0; i < m.rows; i++ {
		matrix.es[i] = *m.At(i, col)
	}
	return matrix
}

func printMat(m mat, padding int, name string) {
	fmt.Printf("%*s%s = [\n", padding, "", name)
	for i := 0; i < m.rows; i++ {
		fmt.Printf("%*s     ", padding, "")
		for j := 0; j < m.cols; j++ {
			fmt.Printf("%.4f ", *m.At(i, j))
		}
		fmt.Printf("\n")
	}
	fmt.Printf("%*s]\n\n", padding, "")
}

func matSig(m *mat) *mat {
	m.es = utils.Map(m.es, sigmoid)
	return m
}

func matReLU(m *mat) *mat {
	m.es = utils.Map(m.es, leakyReLU)
	return m
}

func matFill(m *mat, val float64) *mat {
	//Uses a simple lambda function to return whatever value is passed in matFill and combines that
	//with our newly-made Map function
	m.es = utils.Map(m.es, func(float64) float64 { return val })
	return m
}

func matRand(m *mat, high, low float64) *mat {
	m.es = utils.Map(m.es, func(float64) float64 { return randFloat()*(high-low) + low })
	return m
}

// NOTE: ADDED IN SUM WHEN IT WAS NOT IN nn.h, POTENTIALLY MAKING THIS INCORRECT FOR REASONS I DO
// NOT UNDERSTAND
func matDot(a, b *mat) *mat {
	assert.Assert(a.cols == b.rows)
	dst := mat{
		rows:   a.rows,
		cols:   b.cols,
		stride: b.cols,
		es:     make([]float64, a.rows*b.cols),
	}
	for i := 0; i < dst.rows; i++ {
		for j := 0; j < dst.cols; j++ {
			sum := 0.0
			for k := 0; k < a.cols; k++ {
				sum += *a.At(i, k) * *b.At(k, j)
			}
			*dst.At(i, j) = sum
		}
	}
	return &dst
}

func matSum(dst, src *mat) *mat {
	assert.Assert(dst.cols == src.cols)
	assert.Assert(dst.rows == src.rows)
	for i := 0; i < dst.rows; i++ {
		for j := 0; j < dst.cols; j++ {
			*dst.At(i, j) += *src.At(i, j)
		}
	}
	return dst
}

func matDiff(dst, src *mat) *mat {
	assert.Assert(dst.cols == src.cols)
	assert.Assert(dst.rows == src.rows)
	for i := 0; i < dst.cols; i++ {
		for j := 0; j < dst.rows; j++ {
			*dst.At(i, j) -= *src.At(i, j)
		}
	}
	return dst
}

func matCopy(dst, src *mat) {
	assert.Assert(dst.rows == src.rows)
	assert.Assert(dst.cols == src.cols)
	copy(dst.es, src.es)
}

func matScale(m *mat, scalar float64) {
	utils.MapInPlace(m.es, func(x float64) float64 { return x * scalar })
}

func matAlloc(rows, cols int) mat {
	return mat{
		rows:   rows,
		cols:   cols,
		stride: cols,
		es:     make([]float64, rows*cols),
	}
}

func nnAlloc(arch []int) *nn {
	layers := len(arch)
	assert.Assert(layers > 0)

	net := nn{
		//the count is layers-1 because the input layer is not part of the neural net
		count:   layers - 1,
		weights: make([]mat, layers-1),
		biases:  make([]mat, layers-1),
		//the activations are equal to the number of layers in the architecture because it must
		//include the input
		activations: make([]mat, layers),
	}

	//The input layer to the neural network, coming from the first entry in the arch array.
	net.activations[0] = matAlloc(1, arch[0])

	//Loop through the architecture. We are offset by one in the wieghts and biases because our
	//activations have one additional activation (for the input) and because weights depend on the
	//prior activations.
	for i := 1; i < layers; i++ {
		//The number of columns in the prior activations is defined as the number of neurons in the
		//prior layer. Every neuron in the prior layer is connected to every neuron in the current
		//layer, so our weights matrix needs an entry for each neuron in the prior layer to each
		//in the current. In other words, we need a matrix with as[i-1].cols x arch[i]
		net.weights[i-1] = matAlloc(net.activations[i-1].cols, arch[i])
		net.biases[i-1] = matAlloc(1, arch[i])
		net.activations[i] = matAlloc(1, arch[i])
	}

	return &net
}

func nnPrint(net *nn, name string) {
	namebuff := ""

	fmt.Printf("%v = [\n", name)

	for i := 0; i < net.count; i++ {
		namebuff = fmt.Sprintf("ws%v", i)
		printMat(net.weights[i], 4, namebuff)
		namebuff = fmt.Sprintf("bs%v", i)
		printMat(net.biases[i], 4, namebuff)
	}
	fmt.Printf("]\n")
}

func nnRand(net *nn, high, low float64) {
	for i := 0; i < net.count; i++ {
		matRand(&net.weights[i], high, low)
		matRand(&net.biases[i], high, low)
	}
}

func nnFill(net *nn, val float64) {
	for i := range net.count {
		matFill(&net.weights[i], val)
		matFill(&net.biases[i], val)
		matFill(&net.activations[i], val)
	}
	matFill(&net.activations[net.count], val)
}

func nnForward(net *nn) {
	for i := 0; i < net.count; i++ {
		net.activations[i+1] = *matDot(&net.activations[i], &net.weights[i])
		matSum(&net.activations[i+1], &net.biases[i])
		matReLU(&net.activations[i+1])
	}
}

func nnCost(net nn, ti, to *mat) float64 {
	assert.Assert(ti.rows == to.rows)
	assert.Assert(to.cols == net.activations[net.count].cols)

	cost := 0.0
	for i := range ti.rows {
		x := matRow(ti, i)
		y := matRow(to, i)
		matCopy(&net.activations[0], x)
		nnForward(&net)
		for j := 0; j < to.cols; j++ {
			d := *(net.activations[net.count]).At(0, j) - *y.At(0, j)
			cost += d * d
		}
	}

	return cost / float64(ti.rows)
}

func nnBackProp(net, g *nn, ti, to *mat, clip float64) {
	assert.Assert(ti.rows == to.rows)
	assert.Assert(net.activations[net.count].cols == to.cols)

	nnFill(g, 0.0)

	for i := range ti.rows {
		matCopy(&net.activations[0], matRow(ti, i))
		nnForward(net)
		for j := 0; j <= net.count; j++ {
			matFill(&g.activations[j], 0.0)
		}
		for j := range to.cols {
			*(g.activations[g.count]).At(0, j) = *net.activations[g.count].At(0, j) - *to.At(i, j)
		}

		for l := net.count; l > 0; l-- {
			for j := 0; j < net.activations[l].cols; j++ {
				a := *net.activations[l].At(0, j)
				da := *g.activations[l].At(0, j)

				//*g.biases[l-1].At(0, j) += 2 * da * a * (1 - a)
				grad := 2 * da * derivLeakyReLU(a)
				gradB := clipGradient(grad, clip)
				*g.biases[l-1].At(0, j) += gradB

				for k := 0; k < net.activations[l-1].cols; k++ {
					pa := *net.activations[l-1].At(0, k)
					w := *net.weights[l-1].At(k, j)

					//*g.weights[l-1].At(k, j) += 2 * da * a * (1 - a) * pa
					gradW := grad * pa
					gradW = clipGradient(gradW, clip)
					*g.weights[l-1].At(k, j) += gradW
					//*g.activations[l-1].At(0, k) += 2 * da * a * (1 - a) * w
					gradA := grad * w
					gradA = clipGradient(gradA, clip)
					*g.activations[l-1].At(0, k) += gradA
				}
			}
		}
	}

	for i := 0; i < g.count; i++ {
		for j := 0; j < g.weights[i].rows; j++ {
			for k := 0; k < g.weights[i].cols; k++ {
				*g.weights[i].At(j, k) /= float64(ti.rows)
			}
		}

		for j := 0; j < g.biases[i].rows; j++ {
			for k := 0; k < g.biases[i].cols; k++ {
				*g.biases[i].At(j, k) /= float64(ti.rows)
			}
		}
	}

}

func nnLearn(net, g *nn, rate float64, normalizationValue float64) {
	for i := 0; i < net.count; i++ {
		for j := 0; j < net.weights[i].rows; j++ {
			for k := 0; k < net.weights[i].cols; k++ {
				*net.weights[i].At(j, k) -= rate*(*g.weights[i].At(j, k)) + normalizationValue*(*net.weights[i].At(j, k))
			}
		}
		for j := 0; j < net.biases[i].rows; j++ {
			for k := 0; k < net.biases[i].cols; k++ {
				*net.biases[i].At(j, k) -= rate*(*g.biases[i].At(j, k)) + normalizationValue*(*net.biases[i].At(j, k))
			}
		}
	}
}

func nnGlorotInit(net *nn) {
	for i := range net.count {
		limit := math.Sqrt(6.0 / float64(net.weights[i].cols+net.weights[i].rows))
		net.weights[i].es = utils.Map(net.weights[i].es, func(float64) float64 { return randFloat()*2*limit - limit })
		// limit = math.Sqrt(6.0 / float64(net.biases[i].cols+net.biases[i].rows))
		// net.biases[i].es = utils.Map(net.biases[i].es, func(float64) float64 { return randFloat()*2*limit - limit })
		net.biases[i].es = utils.Map(net.biases[i].es, func(float64) float64 { return 0 })
	}

}

func clipGradient(value, clipValue float64) float64 {
	if value > clipValue {
		return clipValue
	} else if value < -clipValue {
		return -clipValue
	}
	return value
}

var NUMS = 20

func main() {

	n := NUMS
	rows := n * n
	ti := matAlloc(rows, 2)
	to := matAlloc(rows, 1)

	count := 0
	for i := 1; i <= NUMS; i++ {
		for j := 1; j <= NUMS; j++ {
			z := i + j
			*ti.At(count, 0) = float64(i)
			*ti.At(count, 1) = float64(j)
			*to.At(count, 0) = float64(z)
			count++
		}
	}

	printMat(ti, 4, "ti")
	printMat(to, 4, "to")

	arch := []int{2, 10, 1}
	net := nnAlloc(arch)
	g := nnAlloc(arch)
	clipValue := 0.10
	lambda := 0.0
	nnGlorotInit(net)

	matCopy(&net.activations[0], matRow(&ti, 1))
	nnForward(net)
	nnPrint(net, "net")
	initialCost := nnCost(*net, &ti, &to)

	rate := 1e-4

	i := 0
	cost := nnCost(*net, &ti, &to)
	for cost > 1e-5 {
		nnBackProp(net, g, &ti, &to, clipValue)
		nnLearn(net, g, rate, lambda)
		fmt.Printf("%v: cost = %v\n", i, nnCost(*net, &ti, &to))
		i++
		cost = nnCost(*net, &ti, &to)
	}

	finalCost := nnCost(*net, &ti, &to)
	nnPrint(net, "final net")
	fmt.Printf("cost = %v\n", initialCost)
	fmt.Printf("final cost = %v\n", finalCost)
	fmt.Printf("difference in init and final cost = %v\n", initialCost-finalCost)

	for x := 0; x < NUMS*20; x++ {
		for y := 0; y < NUMS*20; y++ {
			*net.activations[0].At(0, 0) = float64(x + 1)
			*net.activations[0].At(0, 1) = float64(y + 1)
			nnForward(net)
			fmt.Printf("%v + %v = %v\n", x+1, y+1, *net.activations[net.count].At(0, 0))
		}
	}

	// fails := 0

	// for x := 0; x < n; x++ {
	// 	for y := 0; y < n; y++ {
	// 		z := x + y

	// 		for j := 0; j < BITS; j++ {
	// 			*net.activations[0].At(0, j) = float64((x >> j) & 1)
	// 			*net.activations[0].At(0, j+BITS) = float64((y >> j) & 1)
	// 		}
	// 		nnForward(net)
	// 		if *net.activations[net.count].At(0, BITS) > 0.5 {
	// 			if z < n {
	// 				fmt.Printf("%v + %v = (OVERFLOW <> %v)\n", x, y, z)
	// 				fails += 1
	// 			}
	// 		} else {
	// 			a := 0
	// 			bit := 0
	// 			for j := 0; j < BITS; j++ {
	// 				if *net.activations[net.count].At(0, j) > 0.5 {
	// 					bit = 1
	// 				} else {
	// 					bit = 0
	// 				}

	// 				a |= bit << j
	// 			}

	// 			if z != a {
	// 				fmt.Printf("%v + %v = (%v <> %v)\n", x, y, z, a)
	// 			}
	// 		}
	// 	}
	// }
	// if fails == 0 {
	// 	fmt.Printf("OK\n")
	// }

}
