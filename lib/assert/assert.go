package assert

func Assert(condition bool, message ...string) {
	if !condition {
		if len(message) > 0 {
			panic(message[0])
		} else {
			panic("Assert failed")
		}
	}
}

func Equal[T comparable](a, b T) {
	if a != b {
		panic("panic: not equal")
	}
}

func NotEqual[T comparable](a, b T) {
	if a == b {
		panic("panic: equal")
	}
}
