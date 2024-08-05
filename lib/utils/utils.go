package utils

type Number interface {
	int | int8 | int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64 | uintptr | float32 | float64
}

func Map[T any, U any](input []T, f func(T) U) []U {
	output := make([]U, len(input))
	for i, v := range input {
		output[i] = f(v)
	}
	return output
}

// MapInPlace *must* map T to T because slices are forced to contain a single type. Therefore we
// must give up some generality and constain our function space to T->T.
func MapInPlace[T any](input []T, f func(T) T) {
	for i := range input {
		input[i] = f(input[i])
	}
}

//go:inline
func MinInt[T Number](a, b T) T {
	if a <= b {
		return a
	} else {
		return b
	}
}
