package facenet

import (
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type Detector struct {
	graph   *tf.Graph
	session *tf.Session

	minSize         float64
	scaleFactor     float64
	scoreThresholds []float32
}

func LoadGraph(modelFile string) (graph *tf.Graph, err error) {
	model, err := ioutil.ReadFile(modelFile)
	if err != nil {
		return nil, err
	}

	graph = tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return nil, err
	}
	return
}

func NewDetector(graph *tf.Graph) (*Detector, error) {
	det := &Detector{minSize: 112, scaleFactor: 0.709, scoreThresholds: []float32{0.6, 0.7, 0.7}}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}

	det.graph = graph
	det.session = session

	return det, nil
}

func (det *Detector) Close() {
	if det.session != nil {
		det.session.Close()
		det.session = nil
	}
}

func (det *Detector) Config(scaleFactor, minSize float64, scoreThresholds []float32) {
	if scaleFactor > 0 {
		det.scaleFactor = scaleFactor
	}
	if minSize > 0 {
		det.minSize = minSize
	}
	if scoreThresholds != nil {
		det.scoreThresholds = scoreThresholds
	}
}

// ComputeFaceFeature 人脸256特征点计算
func (det *Detector) ComputeFaceFeature(tensor *tf.Tensor) ([]float32, error) {
	session := det.session
	graph := det.graph

	img, err := resizeImage(tensor, nil)
	if err != nil {
		return nil, err
	}
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("data").Output(0): img,
		},
		[]tf.Output{
			graph.Operation("fc1_moving_var").Output(0),
		},
		nil)
	if err != nil {
		return nil, err
	}
	facevector := output[0].Value().([][]float32)[0]
	return facevector, err

}

// OptimizeFace 106人脸特征点正脸优选
func (det *Detector) OptimizeFace(tensor *tf.Tensor) ([]float32, error) {
	session := det.session
	graph := det.graph

	img, err := resizeImage(tensor, nil)
	if err != nil {
		return nil, err
	}
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("data").Output(0): img,
		},
		[]tf.Output{
			graph.Operation("landmark_target").Output(0),
		},
		nil)
	if err != nil {
		return nil, err
	}
	facevector := output[0].Value().([][]float32)[0]
	return facevector, err

}

func resizeImage(img *tf.Tensor, scale []float64) (*tf.Tensor, error) {

	var h int32 = 112
	var w int32 = 112

	s := op.NewScope()
	pimg := op.Placeholder(s, tf.Float, op.PlaceholderShape(tf.MakeShape(1, -1, -1, 3)))

	out := op.ResizeBilinear(s, pimg, op.Const(s.SubScope("size"), []int32{h, w}))
	//out = normalizeImage(s, out)

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{pimg: img}, []tf.Output{out})
	if err != nil {
		return nil, err
	}

	return outs[0], nil
}

func normalizeImage(s *op.Scope, input tf.Output) tf.Output {
	out := op.Mul(s, op.Sub(s, input, op.Const(s.SubScope("mean"), float32(127.5))),
		op.Const(s.SubScope("scale"), float32(0.0078125)))
	//out = op.Transpose(s, out, op.Const(s.SubScope("perm"), []int32{0, 2, 1, 3}))
	return out
}

func runScope(s *op.Scope, inputs map[tf.Output]*tf.Tensor, outputs []tf.Output) ([]*tf.Tensor, error) {
	graph, err := s.Finalize()
	if err != nil {
		return nil, err
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	return session.Run(inputs, outputs, nil)
}

func TensorFromJpeg(bytes []byte) (*tf.Tensor, error) {
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		return nil, err
	}

	s := op.NewScope()
	input := op.Placeholder(s, tf.String)
	out := op.ExpandDims(s,
		op.Cast(s, op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
		op.Const(s.SubScope("make_batch"), int32(0)))

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{input: tensor}, []tf.Output{out})
	if err != nil {
		return nil, err
	}

	return outs[0], nil
}
