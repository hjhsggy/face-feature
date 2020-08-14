package attributes

import (
	"bytes"
	"image"
	"image/jpeg"
	"log"

	estimate "face-feature/algorithm/estimater"

	"github.com/nfnt/resize"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	graph *tf.Graph
)

// Init 初始化人脸特征点
func Init(model string) (err error) {
	graph, err = estimate.LoadGraph(model)
	if err != nil {
		log.Fatal(err)
		return
	}
	return
}

// GetFaceAttributes 获取人脸属性
func GetFaceAttributes(bs []byte) (int64, int64, error) {

	// 初始化人脸属性模型
	esti, err := estimate.NewEstimater(graph)
	if err != nil {
		log.Fatal(err)
		return -1, 0, err
	}
	defer esti.Close()

	// resize 输入的人脸图片 64 64
	im, _, err := image.Decode(bytes.NewReader(bs))
	if err != nil {
		return -1, 0, err
	}

	m := resize.Resize(64, 64, im, resize.Lanczos3)
	buf := new(bytes.Buffer)
	err = jpeg.Encode(buf, m, nil)
	target := buf.Bytes()

	// 输入的人脸图片转化为tensor
	imgTensor, err := estimate.TensorFromJpeg(target)
	if err != nil {
		log.Fatal(err)
		return -1, 0, err
	}

	estiGender, estiAge, err := esti.EstimateFaces(imgTensor)

	return estiGender, estiAge, nil
}
