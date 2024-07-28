package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"liz/assert"
	"liz/utils"
	"math"
	"math/rand"
	"os"
)

const (
	version = 0.2
)

// Matrix construct. Stride is number of elements that must be skipped to get to the next row, which
// is often equal to the cols, but not always (especially when decomposing training data into inputs
// and outputs), and es ("elements") is our array of values.
type mat struct {
	rows   uint64
	cols   uint64
	stride uint64
	es     []float64
}

// our neural net struct. Count is the number of layers, activations holds the activations given by
// forwarding the input values (in the 0th entry of the activations) through the neural net. weights
// and biases are the matrices holding the values of the wieghts and biases on each layer.
type nn struct {
	count        uint64
	architecture []uint64
	activations  []mat //activations will be count+1 because we must load the initial values.
	weights      []mat
	biases       []mat
}

type DiffVersionWarning struct {
	current float64
	given   float64
}

func (e *DiffVersionWarning) Error() string {
	return fmt.Sprintf("Difference in version, expected %v, got %v", e.current, e.given)
}

// gives us a pointer to the value at a specific row and column of a matrix.
func (m *mat) At(row, col uint64) *float64 {
	return &m.es[row*m.stride+col]
}

// initializes a new matrix with the specified values and returns it.
func newMat(r, c, s uint64, e []float64) mat {
	var m mat = mat{
		rows:   r,
		cols:   c,
		stride: s,
		es:     e,
	}
	return m
}

//------------------ Cost function for a perceptron, now obsolete ----------------------------
// Calculates the cost of our two weights and bias, wanting as close as 0 as possible.
// func cost(w1, w2, b float64, m *mat) float64 {
// 	result := 0.0
// 	for i := 0; i < m.rows; i++ {
// 		x1 := *m.At(i, 0)
// 		x2 := *m.At(i, 1)
// 		y := sigmoid(x1*w1 + x2*w2 + b)
// 		d := y - *m.At(i, 2)
// 		result += d * d
// 	}
// 	return result / float64(m.rows)
// }
//---------------------------------------------------------------------------------------------

//------------ Function to compute the gradient of a perception, now obsolete ------------------
// func gcost(w1, w2, b float64, m *mat) (dw1, dw2, db float64) {
// 	dw1, dw2, db = 0, 0, 0
// 	for i := 0; i < m.rows; i++ {
// 		xi := *m.At(i, 0)
// 		yi := *m.At(i, 1)
// 		zi := *m.At(i, 2)
// 		ai := sigmoid(w1*xi + w2*yi + b)
// 		di := 2 * (ai - zi) * ai * (1 - ai)

// 		dw1 += di * xi
// 		dw2 += di * yi
// 		db += di
// 	}
// 	dw1 /= float64(m.rows)
// 	dw2 /= float64(m.rows)
// 	db /= float64(m.rows)

// 	return dw1, dw2, db
// }
//----------------------------------------------------------------------------------------------

// alias for the rand.Float64() function
func randFloat() float64 {
	return rand.Float64()
}

// Computes the sigmoid of x using the definition of the sigmoid function
func sigmoid(x float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-x))
}

// Computes the result of applying the Leaky ReLU function to x using the definition of the Leakey ReLU
func leakyReLU(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0.01 * x
	}
}

// Returns the derivative of the Leaky ReLU, for ease of understanding when writing the equations later.
func derivLeakyReLU(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0.01
	}
}

// Returns a single row from the matrix by finding the start of the elements and selecting the slice
// representing all elements belonging to that row
func matRow(m *mat, row uint64) *mat {
	start := row * m.stride
	return &mat{
		rows:   1,
		cols:   m.cols,
		stride: m.stride,
		es:     m.es[start : start+m.cols],
	}
}

// Returns a single column from the matrix by creating a new element slice and adding elements one by
// one from the original elements.
func matCol(m *mat, col uint64) mat {
	matrix := mat{
		rows:   m.rows,
		cols:   1,
		stride: 1,
		es:     make([]float64, m.rows),
	}
	for i := uint64(0); i < m.rows; i++ {
		matrix.es[i] = *m.At(i, col)
	}
	return matrix
}

// Prints a matrix by printing out padding, then printing out each element in the matrix.
func printMat(m mat, padding int, name string) {
	fmt.Printf("%*s%s = [\n", padding, "", name)
	for i := uint64(0); i < m.rows; i++ {
		fmt.Printf("%*s     ", padding, "")
		for j := uint64(0); j < m.cols; j++ {
			fmt.Printf("%.4f ", *m.At(i, j))
		}
		fmt.Printf("\n")
	}
	fmt.Printf("%*s]\n\n", padding, "")
}

// Applies the sigmoid function to a matrix using the Map function from my utils library.
func matSig(m *mat) *mat {
	m.es = utils.Map(m.es, sigmoid)
	return m
}

// Applies the Leaky ReLU function to a matrix using the Map function from my utils library.
func matReLU(m *mat) *mat {
	m.es = utils.Map(m.es, leakyReLU)
	return m
}

// Uses a simple lambda function to return whatever value is passed in matFill and combines that
// with our newly-made Map function
func matFill(m *mat, val float64) *mat {
	m.es = utils.Map(m.es, func(float64) float64 { return val })
	return m
}

// Simply randomizes a matrix using a lambda function and a map function, but it is now obsolete
// because we use nnGlorotInit to seed our neural nets.
func matRand(m *mat, high, low float64) *mat {
	m.es = utils.Map(m.es, func(float64) float64 { return randFloat()*(high-low) + low })
	return m
}

// Computes the dot product of two matrices using the definition of the dot product
func matDot(a, b *mat) *mat {
	assert.Assert(a.cols == b.rows)
	dst := mat{
		rows:   a.rows,
		cols:   b.cols,
		stride: b.cols,
		es:     make([]float64, a.rows*b.cols),
	}
	for i := uint64(0); i < dst.rows; i++ {
		for j := uint64(0); j < dst.cols; j++ {
			sum := 0.0
			for k := uint64(0); k < a.cols; k++ {
				sum += *a.At(i, k) * *b.At(k, j)
			}
			*dst.At(i, j) = sum
		}
	}
	return &dst
}

// Adds the source matrix to the destination matrix.
func matSum(dst, src *mat) *mat {
	assert.Assert(dst.cols == src.cols)
	assert.Assert(dst.rows == src.rows)
	for i := uint64(0); i < dst.rows; i++ {
		for j := uint64(0); j < dst.cols; j++ {
			*dst.At(i, j) += *src.At(i, j)
		}
	}
	return dst
}

// Subtracts the source matrix from the destination matrix.
func matDiff(dst, src *mat) *mat {
	assert.Assert(dst.cols == src.cols)
	assert.Assert(dst.rows == src.rows)
	for i := uint64(0); i < dst.cols; i++ {
		for j := uint64(0); j < dst.rows; j++ {
			*dst.At(i, j) -= *src.At(i, j)
		}
	}
	return dst
}

// Compies the data from the source to the destination matrix.
func matCopy(dst, src *mat) {
	assert.Assert(dst.rows == src.rows)
	assert.Assert(dst.cols == src.cols)
	copy(dst.es, src.es)
}

// Uses the MapInPlace function from my utils library to scale all values in the matrix.
func matScale(m *mat, scalar float64) {
	utils.MapInPlace(m.es, func(x float64) float64 { return x * scalar })
}

// Allocates a matrix with the given parameters and returns it
func matAlloc(rows, cols uint64) mat {
	return mat{
		rows:   rows,
		cols:   cols,
		stride: cols,
		es:     make([]float64, rows*cols),
	}
}

// Allocates a neural network with the given architecture.
func nnAlloc(arch []uint64) *nn {
	layers := uint64(len(arch))
	assert.Assert(layers > 0)

	net := nn{
		//the count is layers-1 because the input layer is not part of the neural net
		count:        layers - 1,
		architecture: arch,
		weights:      make([]mat, layers-1),
		biases:       make([]mat, layers-1),
		//the activations are equal to the number of layers in the architecture because it must
		//include the input
		activations: make([]mat, layers),
	}

	//The input layer to the neural network, coming from the first entry in the arch array.
	net.activations[0] = matAlloc(1, arch[0])

	//Loop through the architecture. We are offset by one in the wieghts and biases because our
	//activations have one additional activation (for the input) and because weights depend on the
	//prior activations.
	for i := uint64(1); i < layers; i++ {
		//The number of columns in the prior activations is defined as the number of neurons in the
		//prior layer. Every neuron in the prior layer is connected to every neuron in the current
		//layer, so our weights matrix needs an entry for each neuron in the prior layer to each
		//in the current. In other words, we need a matrix of the form as[i-1].cols x arch[i]
		net.weights[i-1] = matAlloc(net.activations[i-1].cols, arch[i])
		net.biases[i-1] = matAlloc(1, arch[i])
		net.activations[i] = matAlloc(1, arch[i])
	}

	return &net
}

// Prints a neural network by printing each of its constitutent matrices
func nnPrint(net *nn, name string) {
	namebuff := ""

	fmt.Printf("%v = [\n", name)

	for i := uint64(0); i < net.count; i++ {
		namebuff = fmt.Sprintf("ws%v", i)
		printMat(net.weights[i], 4, namebuff)
		namebuff = fmt.Sprintf("bs%v", i)
		printMat(net.biases[i], 4, namebuff)
	}
	fmt.Printf("]\n")
}

// Randomizes a neural network naiively. This is not deprecated because nnGlorotInit is a superior
// way to initialize a neural network.
func nnRand(net *nn, high, low float64) {
	for i := uint64(0); i < net.count; i++ {
		matRand(&net.weights[i], high, low)
		matRand(&net.biases[i], high, low)
	}
}

// Fills a neutral network with the specified value, usually 0.
func nnFill(net *nn, val float64) {
	for i := range net.count {
		matFill(&net.weights[i], val)
		matFill(&net.biases[i], val)
		matFill(&net.activations[i], val)
	}
	//activations has one extra layer for the input, so we have to make sure to fill one final
	//layer.
	matFill(&net.activations[net.count], val)
}

// Forwards values through a neural network by computing the dot product of the current layer's
// activations and weights and stores it in the next layer's activation, then applies the bias. Finally
// it applies the ReLU function.
func nnForward(net *nn) {
	for i := uint64(0); i < net.count; i++ {
		net.activations[i+1] = *matDot(&net.activations[i], &net.weights[i])
		matSum(&net.activations[i+1], &net.biases[i])
		matReLU(&net.activations[i+1])
	}
}

// Computes the cost of the network by taking in the training input (ti) and training output (to),
// forwarding the input through the network, and comparing the output of the neural net to the
// actual output. It squares this difference and adds it to the cost.
func nnCost(net nn, ti, to *mat) float64 {
	assert.Assert(ti.rows == to.rows)
	assert.Assert(to.cols == net.activations[net.count].cols)

	cost := 0.0
	for i := range ti.rows {
		x := matRow(ti, i)
		y := matRow(to, i)
		matCopy(&net.activations[0], x)
		nnForward(&net)
		for j := uint64(0); j < to.cols; j++ {
			d := *(net.activations[net.count]).At(0, j) - *y.At(0, j)
			cost += d * d
		}
	}

	return cost / float64(ti.rows)
}

// Back propegates the neural network by computing the partial derivatives of the weights and biases
// and adds them to the gradient so that they can be applied during the learn step.
func nnBackProp(net, g *nn, ti, to *mat, clip float64) {
	assert.Assert(ti.rows == to.rows)
	assert.Assert(net.activations[net.count].cols == to.cols)

	nnFill(g, 0.0)

	//For each row in the training input, we have to compute the partial derivatives and add it to
	//the gradients.
	for i := range ti.rows {

		//We copy in the inputs and forward it so we have something to calculate based off of.
		matCopy(&net.activations[0], matRow(ti, i))
		nnForward(net)

		//We will the activations in our gradient with 0s because we'll be using them each loop to
		//store our intermediate values.
		for j := uint64(0); j <= net.count; j++ {
			matFill(&g.activations[j], 0.0)
		}

		//We store the difference between our predicted output and the actual output in the
		//activations
		for j := range to.cols {
			*(g.activations[g.count]).At(0, j) = *net.activations[g.count].At(0, j) - *to.At(i, j)
		}

		//We loop through each layer starting from the last and moving backwards because the partial
		//derivatives must be computed based on the derivatives that in the next layer. Note that
		//we stop at layer 1 because we always set the prior layer.

		//Activations in the gradient aren't used right now, so we can temporarily use them to store
		//our derivatives in order to not use more memory.

		//l = current layer, though we always compute weights/biases for the layer before
		//j = current column in current layer
		//k = current column in previous layer

		for l := net.count; l > 0; l-- {
			for j := uint64(0); j < net.activations[l].cols; j++ {
				//a = the activation that we have
				//da = the difference between the activation we have and the output
				a := *net.activations[l].At(0, j)
				da := *g.activations[l].At(0, j)

				//grad will be used in all other gradients afterwards, so we set it up now as the
				//partial derivative (which is also the partial derivative with respect to the biases)
				grad := 2 * da * derivLeakyReLU(a)
				//We clip the gradient before setting it and adding it to the gradient bias matrix
				//to ensure the value doesn't erratically change
				gradB := clipGradient(grad, clip)
				*g.biases[l-1].At(0, j) += gradB

				for k := uint64(0); k < net.activations[l-1].cols; k++ {
					//grab the partial activation from the previous layer because it appears in the
					//partial derivative of the weights. we only ever need it to be the 0th row
					//becuase there's only ever one activation per column.
					pa := *net.activations[l-1].At(0, k)

					//grab the weight from the prior layer because it appears in the partial
					//derivative of the activations
					w := *net.weights[l-1].At(k, j)

					//multiplies the grad calculation by the prior activation to complete the formula
					//for the partial derivative of the weights, then clips it, then puts it into
					//the gradient under the prior layer's weights.
					gradW := grad * pa
					gradW = clipGradient(gradW, clip)
					*g.weights[l-1].At(k, j) += gradW

					//multiplies the grad calculation by the weights to complete the formula for the
					//partial derivative of the activations, then clips it, then puts it into the
					//gradient under the prior layer's weights.
					gradA := grad * w
					gradA = clipGradient(gradA, clip)
					*g.activations[l-1].At(0, k) += gradA
				}
			}
		}
	}

	//Averages the biases and weights over the number of training examples. Could possibly be done
	//with matScale more efficiently/cleanly?
	for i := uint64(0); i < g.count; i++ {
		for j := uint64(0); j < g.weights[i].rows; j++ {
			for k := uint64(0); k < g.weights[i].cols; k++ {
				*g.weights[i].At(j, k) /= float64(ti.rows)
			}
		}

		for j := uint64(0); j < g.biases[i].rows; j++ {
			for k := uint64(0); k < g.biases[i].cols; k++ {
				*g.biases[i].At(j, k) /= float64(ti.rows)
			}
		}
	}

}

// applies the gradient to the neural network by multiplying it by the rate and then potentially
// adding the current value of the weights and biases multiplied by the normalization value.
func nnLearn(net, g *nn, rate float64, normalizationValue float64) {
	for i := uint64(0); i < net.count; i++ {
		for j := uint64(0); j < net.weights[i].rows; j++ {
			for k := uint64(0); k < net.weights[i].cols; k++ {
				*net.weights[i].At(j, k) -= rate*(*g.weights[i].At(j, k)) + normalizationValue*(*net.weights[i].At(j, k))
			}
		}
		for j := uint64(0); j < net.biases[i].rows; j++ {
			for k := uint64(0); k < net.biases[i].cols; k++ {
				*net.biases[i].At(j, k) -= rate*(*g.biases[i].At(j, k)) + normalizationValue*(*net.biases[i].At(j, k))
			}
		}
	}
}

// Uses the Glorot function to initialize weights to random values
func nnGlorotInit(net *nn) {
	for i := range net.count {
		limit := math.Sqrt(6.0 / float64(net.weights[i].cols+net.weights[i].rows))
		net.weights[i].es = utils.Map(net.weights[i].es, func(float64) float64 { return randFloat()*2*limit - limit })
		net.biases[i].es = utils.Map(net.biases[i].es, func(float64) float64 { return 0 })
	}

}

// Clips gradient values to ensure that gradients don't soar into infinity
func clipGradient(value, clipValue float64) float64 {
	if value > clipValue {
		return clipValue
	} else if value < -clipValue {
		return -clipValue
	}
	return value
}

func optimize_adam(net, g, m, v *nn, rate, epsilon, beta1, beta2 float64, epoch uint64) {

	epoch_normalized := math.Max(float64(epoch), 1)

	for i := range g.count {
		for j := range g.weights[i].rows {
			for k := range g.weights[i].cols {

				moment := m.weights[i].At(j, k)
				velocity := v.weights[i].At(j, k)
				gradient := g.weights[i].At(j, k)
				network := net.weights[i].At(j, k)

				*moment = beta1**moment + (1-beta1)**gradient
				*velocity = beta2**velocity + (1-beta2)*(*gradient**gradient)

				m_hat := *moment / (1 - math.Pow(beta1, epoch_normalized))
				v_hat := *velocity / (1 - math.Pow(beta2, epoch_normalized))

				*network = *network - (rate * m_hat / (math.Sqrt(v_hat) + epsilon))
			}
		}

		for j := range g.biases[i].rows {
			for k := range g.biases[i].cols {
				moment := m.biases[i].At(j, k)
				velocity := v.biases[i].At(j, k)
				gradient := g.biases[i].At(j, k)
				network := net.biases[i].At(j, k)

				*moment = beta1**moment + (1-beta1)**gradient
				*velocity = beta2**velocity + (1-beta2)*(*gradient**gradient)

				m_hat := *moment / (1 - math.Pow(beta1, epoch_normalized))
				v_hat := *velocity / (1 - math.Pow(beta2, epoch_normalized))

				*network = *network - (rate * m_hat / (math.Sqrt(v_hat) + epsilon))
			}
		}
	}
}

func exportNeuralNetworkBinary(net *nn, path string) error {

	if _, err := os.Stat(path); err == nil {
		if err := os.Remove(path); err != nil {
			fmt.Printf("Error removing file: %v\n", err)
			return errors.New("error removing file")
		}
	}
	file, err := os.Create(path)

	if err != nil {
		return err
	}
	defer file.Close()

	if err := binary.Write(file, binary.LittleEndian, float64(version)); err != nil {
		return err
	}

	layers := net.count + 1
	if err := binary.Write(file, binary.LittleEndian, uint64(layers)); err != nil {
		return err
	}

	for _, neurons := range net.architecture {
		if err := binary.Write(file, binary.LittleEndian, uint64(neurons)); err != nil {
			return err
		}
	}

	for _, weightMatrix := range net.weights {
		for _, weight := range weightMatrix.es {
			if err := binary.Write(file, binary.LittleEndian, float64(weight)); err != nil {
				return err
			}
		}
	}

	for _, biasMatrix := range net.biases {
		for _, bias := range biasMatrix.es {
			if err := binary.Write(file, binary.LittleEndian, float64(bias)); err != nil {
				return err
			}
		}
	}

	return nil
}

func importNeuralNetworkBinary(path string) (*nn, error) {

	var early *nn
	var returnErr error

	file, err := os.Open(path)
	if err != nil {
		return early, err
	}
	defer file.Close()

	var ver float64
	if err := binary.Read(file, binary.LittleEndian, &ver); err != nil {
		return early, err
	}

	if ver != version {
		fmt.Println("warning: neural network was created in a different version of 'nn.go'. unpredictable errors may arise. execution will continue.")
		returnErr = &DiffVersionWarning{current: version, given: ver}
	}

	var layers uint64
	if err := binary.Read(file, binary.LittleEndian, &layers); err != nil {
		return early, err
	}

	arch := make([]uint64, layers)
	for i := range layers {
		if err := binary.Read(file, binary.LittleEndian, &arch[i]); err != nil {
			return early, err
		}
	}

	net := nnAlloc(arch)

	i, j := 0, 1
	for j < int(layers) {
		reads := arch[i] * arch[j]
		matrix := matAlloc(arch[i], arch[j])

		for k := range reads {
			if err := binary.Read(file, binary.LittleEndian, &matrix.es[k]); err != nil {
				return net, err
			}
		}
		net.weights[i] = matrix
		i++
		j++
	}

	i = 1
	for i < int(layers) {
		reads := arch[i]
		matrix := matAlloc(1, arch[i])
		for k := range reads {
			if err := binary.Read(file, binary.LittleEndian, &matrix.es[k]); err != nil {
				return net, err
			}
		}
		net.biases[i-1] = matrix
		i++
	}

	return net, returnErr
}

var NUMS = 100

func main() {

	n := NUMS
	rows := uint64(n * n)
	ti := matAlloc(rows, 2)
	to := matAlloc(rows, 1)

	count := uint64(0)
	for i := 1; i <= NUMS; i++ {
		for j := 1; j <= NUMS; j++ {
			z := i + j
			*ti.At(count, 0) = float64(i)
			*ti.At(count, 1) = float64(j)
			*to.At(count, 0) = float64(z)
			count++
		}
	}

	// printMat(ti, 4, "ti")
	// printMat(to, 4, "to")

	arch := []uint64{2, 8, 8, 1}
	net := nnAlloc(arch)
	g := nnAlloc(arch)

	nnGlorotInit(net)

	matCopy(&net.activations[0], matRow(&ti, 1))
	nnForward(net)
	nnPrint(net, "net")
	initialCost := nnCost(*net, &ti, &to)

	clipValue := .4
	rate := 1e-3
	beta1 := 0.9
	beta2 := 0.999
	epsilon := 1e-8
	moments := nnAlloc(arch)
	velocities := nnAlloc(arch)

	epoch := uint64(0)
	cost := nnCost(*net, &ti, &to)
	for cost > 1e-5 {
		nnBackProp(net, g, &ti, &to, clipValue)
		//nnLearn(net, g, rate, lambda)
		optimize_adam(net, g, moments, velocities, rate, epsilon, beta1, beta2, epoch)
		fmt.Printf("%v: cost = %v\n", epoch, nnCost(*net, &ti, &to))
		epoch++
		cost = nnCost(*net, &ti, &to)
	}

	finalCost := nnCost(*net, &ti, &to)
	nnPrint(net, "final net")
	fmt.Printf("cost = %v\n", initialCost)
	fmt.Printf("final cost = %v\n", finalCost)
	fmt.Printf("difference in init and final cost = %v\n", initialCost-finalCost)

	exportNeuralNetworkBinary(net, "adder.nn")

	// for x := 0; x < NUMS*20; x++ {
	// 	for y := 0; y < NUMS*20; y++ {
	// 		*net.activations[0].At(0, 0) = float64(x + 1)
	// 		*net.activations[0].At(0, 1) = float64(y + 1)
	// 		nnForward(net)
	// 		fmt.Printf("%v + %v = %v\n", x+1, y+1, *net.activations[net.count].At(0, 0))
	// 	}
}
