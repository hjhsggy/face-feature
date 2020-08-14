package estimate

import (
	"errors"
	"io/ioutil"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Detector Mtcnn检测模型
type Detector struct {
	graph   *tf.Graph
	session *tf.Session
}

// LoadGraph 初始化模型
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

// NewEstimater 年龄检测
func NewEstimater(graph *tf.Graph) (*Detector, error) {
	esti := &Detector{}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}

	esti.graph = graph
	esti.session = session

	return esti, nil
}

func (det *Detector) Close() {
	if det.session != nil {
		det.session.Close()
		det.session = nil
	}
}

// EstimateFaces 性别年龄检测
func (det *Detector) EstimateFaces(tensor *tf.Tensor) (int64, int64, error) {
	session := det.session
	graph := det.graph

	var err error
	//face_size := 64

	// h := float32(tensor.Shape()[1])
	// w := float32(tensor.Shape()[2])
	// log.Println(h, "=======Start to estimate age and gender=======", w)
	// log.Println("tensor.DataType: ", tensor.DataType())

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("input_1").Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation("dense_1/Softmax").Output(0),
			graph.Operation("dense_2/Softmax").Output(0),
		},
		nil)
	if err != nil {
		return 0, 0, err
	}
	// log.Println("output[0] Value: ", output[0].Value().([][]float32)[0])
	// log.Printf("output[1] Value's Type: %T", output[1].Value().([][]float32)[0])

	//GenderScore:[]float32 元素是男的概率和女的概率,长度为2
	//AgeScore:[]float32 元素是岁数的概率,长度为101,从0到100
	genderScoreArray := output[0].Value().([][]float32)[0]
	ageScoreArray := output[1].Value().([][]float32)[0]

	//argmax函数来回归出最大值的序号
	gender, err := RegressMax(genderScoreArray)
	if err != nil {
		return 0, 0, err
	}
	ageScore, err := RegressMax(ageScoreArray)
	if err != nil {
		return 0, 0, err
	}
	// log.Println(AgeScoreArray)

	return gender, ageScore, err
}

// RegressMax 回归最大值
func RegressMax(input []float32) (int64, error) {
	if len(input) == 0 {
		return 0, errors.New("input list has no elements")
	}
	var output int64
	MaxElement := input[output]
	for i := range input {
		if input[i] > MaxElement {
			output = int64(i)
			MaxElement = input[output]
		}
	}
	return output, nil

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

// TensorFromJpeg 图片数据转化为tensor对象
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
