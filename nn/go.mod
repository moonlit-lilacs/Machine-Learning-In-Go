module liz/neuralnet

go 1.22.5

replace liz/assert => ../lib/assert

require (
	liz/assert v0.0.0-00010101000000-000000000000
	liz/utils v0.0.0-00010101000000-000000000000
)

replace liz/utils => ../lib/utils/
