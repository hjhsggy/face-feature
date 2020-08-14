package mtcnn

import (
	"errors"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"math"

	"github.com/nfnt/resize"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Detector Mtcnn检测模型
type Detector struct {
	graph   *tf.Graph
	session *tf.Session

	minSize         float64
	scaleFactor     float64
	scoreThresholds []float32
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

// NewDetector 人脸侦测
func NewDetector(graph *tf.Graph) (*Detector, error) {
	det := &Detector{minSize: 40.0, scaleFactor: 0.709, scoreThresholds: []float32{0.6, 0.7, 0.8}}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}

	det.graph = graph
	det.session = session

	return det, nil
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

// DetectFaces
func (det *Detector) DetectFaces(tensor *tf.Tensor) ([][]float32, [][]float32, []float32, error) {
	session := det.session
	graph := det.graph

	var err error
	var total_bbox, total_reg [][]float32
	var total_score []float32

	h := float32(tensor.Shape()[1])
	w := float32(tensor.Shape()[2])
	scales := scales(float64(h), float64(w), det.scaleFactor, det.minSize)
	// log.Println("scales:", scales)

	// stage 1
	for _, scale := range scales {
		img, err := resizeImage(tensor, scale)
		if err != nil {
			return nil, nil, nil, err
		}
		output, err := session.Run(
			map[tf.Output]*tf.Tensor{
				graph.Operation("pnet/input").Output(0): img,
			},
			[]tf.Output{
				graph.Operation("pnet/conv4-2/BiasAdd").Output(0),
				graph.Operation("pnet/prob1").Output(0),
			},
			nil)
		if err != nil {
			return nil, nil, nil, err
		}

		// log.Println("pnet:", img.Shape(), "=>", output[0].Shape(), ",", output[1].Shape())

		out0, _ := transpose(output[0], []int32{0, 2, 1, 3})
		out1, _ := transpose(output[1], []int32{0, 2, 1, 3})

		xreg := out0.Value().([][][][]float32)[0]
		xscore := out1.Value().([][][][]float32)[0]

		bbox, reg, score := generateBbox(xscore, xreg, scale, det.scoreThresholds[0])
		if len(bbox) == 0 {
			continue
		}

		bbox, reg, score, err = nms(bbox, reg, score, 0.5)
		if len(bbox) > 0 {
			total_bbox = append(total_bbox, bbox...)
			total_reg = append(total_reg, reg...)
			total_score = append(total_score, score...)
		}
	}

	// log.Println("stage 1 bbox:", len(total_bbox))

	if len(total_bbox) == 0 {
		return nil, nil, nil, nil
	}

	total_bbox, total_reg, total_score, err = nms(total_bbox, total_reg, total_score, 0.5)
	// log.Println("stage 1 nms bbox:", len(total_bbox), err)

	if len(total_bbox) == 0 {
		return nil, nil, nil, nil
	}

	//calibrate & square
	for i, box := range total_bbox {
		total_bbox[i] = square(adjustBbox(box, total_reg[i]))
	}

	// stage 2
	imgs, err := cropResizeImage(tensor, normalizeBbox(total_bbox, w, h), []int32{24, 24}, true)
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("rnet/input").Output(0): imgs,
		},
		[]tf.Output{
			graph.Operation("rnet/conv5-2/conv5-2").Output(0),
			graph.Operation("rnet/prob1").Output(0),
		},
		nil)
	if err != nil {
		return nil, nil, nil, err
	}

	// log.Println("rnet:", imgs.Shape(), "=>", output[0].Shape(), ",", output[1].Shape())

	//filter
	reg := output[0].Value().([][]float32)
	score := output[1].Value().([][]float32)
	total_bbox, total_reg, total_score = filterBbox(total_bbox, reg, score, det.scoreThresholds[1])
	// log.Println("stage 2, filter bbox: ", len(total_bbox))

	if len(total_bbox) == 0 {
		return nil, nil, nil, nil
	}

	total_bbox, total_reg, total_score, err = nms(total_bbox, total_reg, total_score, 0.5)
	// log.Println("stage 2, nms bbox: ", len(total_bbox), err)

	if len(total_bbox) == 0 {
		return nil, nil, nil, nil
	}

	//calibrate, square
	for i, box := range total_bbox {
		total_bbox[i] = square(adjustBbox(box, total_reg[i]))
	}

	// stage 3
	imgs, err = cropResizeImage(tensor, normalizeBbox(total_bbox, w, h), []int32{48, 48}, true)
	output, err = session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("onet/input").Output(0): imgs,
		},
		[]tf.Output{
			graph.Operation("onet/conv6-2/conv6-2").Output(0),
			graph.Operation("onet/conv6-3/conv6-3").Output(0),
			graph.Operation("onet/prob1").Output(0),
		},
		nil)
	if err != nil {
		return nil, nil, nil, err
	}

	// log.Println("onet:", imgs.Shape(), "=>", output[0].Shape(), ",", output[1].Shape(), ",", output[2].Shape())

	reg = output[0].Value().([][]float32)
	keypoint := output[1].Value().([][]float32)
	score = output[2].Value().([][]float32)
	total_bbox, total_reg, total_score, keypoint = filterBboxOnet(total_bbox, reg, score, det.scoreThresholds[2], keypoint)
	// log.Println("stage 3, filter bbox: ", len(total_bbox))

	if len(total_bbox) == 0 {
		return nil, nil, nil, nil
	}

	for i, box := range total_bbox {
		total_bbox[i] = adjustBbox(box, total_reg[i])
		//fmt.Printf("执行")
		keypoint[i] = adjustKeypoint(box, keypoint[i])
	}

	//	for i, onefacekp := range keypoint {
	//	keypoint[i] = adjustKeypoint(onefacekp, keypoint[i])
	//}
	keypoint, total_bbox, _, total_score, err = nmsOnet(keypoint, total_bbox, total_reg, total_score, 0.5)
	// log.Println("stage 3, nms bbox: ", len(total_bbox), err)
	//fmt.Printf("执行")
	return total_bbox, keypoint, total_score, nil
}

// EstimateFaces 性别年龄检测
func (det *Detector) EstimateFaces(tensor *tf.Tensor) (int, int, error) {
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

func adjustKeypoint(bbox []float32, regkeypoint []float32) []float32 {
	if len(regkeypoint) == 10 && len(regkeypoint) == 10 {
		w := bbox[2] - bbox[0] + 1.0
		h := bbox[3] - bbox[1] + 1.0
		//fmt.Printf("gg%v hh%v aa%v yy%v gg%v hh%v aa%v yy%v aa%v yy%v\n", regkeypoint[0], regkeypoint[1], regkeypoint[2], regkeypoint[3], regkeypoint[4], regkeypoint[5], regkeypoint[6], regkeypoint[7], regkeypoint[8], regkeypoint[9])
		regkeypoint[0] = bbox[0] + regkeypoint[0]*w
		regkeypoint[1] = bbox[0] + regkeypoint[1]*w
		regkeypoint[2] = bbox[0] + regkeypoint[2]*w
		regkeypoint[3] = bbox[0] + regkeypoint[3]*w
		regkeypoint[4] = bbox[0] + regkeypoint[4]*w
		regkeypoint[5] = bbox[1] + regkeypoint[5]*h
		regkeypoint[6] = bbox[1] + regkeypoint[6]*h
		regkeypoint[7] = bbox[1] + regkeypoint[7]*h
		regkeypoint[8] = bbox[1] + regkeypoint[8]*h
		regkeypoint[9] = bbox[1] + regkeypoint[9]*h

	}
	return regkeypoint
}

func adjustBbox(bbox []float32, reg []float32) []float32 {
	if len(bbox) == 4 && len(reg) == 4 {
		w := bbox[2] - bbox[0] + 1.0
		h := bbox[3] - bbox[1] + 1.0

		bbox[0] += reg[0] * w
		bbox[1] += reg[1] * h
		bbox[2] += reg[2] * w
		bbox[3] += reg[3] * h
	}

	return square(bbox)
}

func square(bbox []float32) []float32 {
	if len(bbox) != 4 {
		return bbox
	}
	w := bbox[2] - bbox[0]
	h := bbox[3] - bbox[1]

	l := w
	if l < h {
		l = h
	}

	bbox[0] = bbox[0] + 0.5*w - 0.5*l
	bbox[1] = bbox[1] + 0.5*h - 0.5*l
	bbox[2] = bbox[0] + l
	bbox[3] = bbox[1] + l
	return bbox
}

func normalizeBbox(bbox [][]float32, w, h float32) [][]float32 {
	out := make([][]float32, len(bbox))
	for i, box := range bbox {
		ibox := make([]float32, 4)
		//NOTE: y1, x1, y2, x2
		ibox[0] = box[1] / h
		ibox[1] = box[0] / w
		ibox[2] = box[3] / h
		ibox[3] = box[2] / w
		out[i] = ibox
	}

	return out
}

func filterBbox(bbox, reg [][]float32, score [][]float32, threshold float32) (nbbox, nreg [][]float32, nscore []float32) {
	for i, x := range score {
		if x[1] > threshold {
			nbbox = append(nbbox, bbox[i])
			nreg = append(nreg, reg[i])
			nscore = append(nscore, x[1])
		}
	}
	return
}

func filterBboxOnet(bbox, reg [][]float32, score [][]float32, threshold float32, keypoint [][]float32) (nbbox, nreg [][]float32, nscore []float32, nkeypoint [][]float32) {
	for i, x := range score {
		if x[1] > threshold {
			nkeypoint = append(nkeypoint, keypoint[i])
			nbbox = append(nbbox, bbox[i])
			nreg = append(nreg, reg[i])
			nscore = append(nscore, x[1])
		}
	}
	return
}

func generateBbox(imap [][][]float32, reg [][][]float32, scale float64, threshold float32) (bbox, nreg [][]float32, score []float32) {
	const (
		Stride   = 2.0
		CellSize = 12.0
	)

	for i, x := range imap {
		for j, y := range x {
			if y[1] > threshold {
				n := []float32{float32(math.Floor((Stride*float64(j)+1.0)/scale + 0.5)),
					float32(math.Floor((Stride*float64(i)+1.0)/scale + 0.5)),
					float32(math.Floor((Stride*float64(j)+1.0+CellSize)/scale + 0.5)),
					float32(math.Floor((Stride*float64(i)+1.0+CellSize)/scale + 0.5)),
				}
				bbox = append(bbox, n)
				nreg = append(nreg, reg[i][j])
				score = append(score, y[1])
			}
		}
	}

	return
}

func nmsOnet(keypoint, bbox, reg [][]float32, score []float32, threshold float32) (nkeypoint, nbbox, nreg [][]float32, nscore []float32, err error) {
	tbbox, _ := tf.NewTensor(bbox)
	tscore, _ := tf.NewTensor(score)

	s := op.NewScope()
	pbbox := op.Placeholder(s.SubScope("bbox"), tf.Float, op.PlaceholderShape(tf.MakeShape(-1, 4)))
	pscore := op.Placeholder(s.SubScope("score"), tf.Float, op.PlaceholderShape(tf.MakeShape(-1)))

	out := op.NonMaxSuppression(s, pbbox, pscore, op.Const(s.SubScope("max_len"), int32(len(bbox))), op.NonMaxSuppressionIouThreshold(threshold))

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{pbbox: tbbox, pscore: tscore}, []tf.Output{out})
	if err != nil {
		return
	}

	pick := outs[0]
	if pick != nil {
		if idx, ok := pick.Value().([]int32); ok {
			for _, i := range idx {
				nkeypoint = append(nkeypoint, keypoint[i])
				nbbox = append(nbbox, bbox[i])
				nreg = append(nreg, reg[i])
				nscore = append(nscore, score[i])
			}
		}
	}

	return
}

func nms(bbox, reg [][]float32, score []float32, threshold float32) (nbbox, nreg [][]float32, nscore []float32, err error) {
	tbbox, _ := tf.NewTensor(bbox)
	tscore, _ := tf.NewTensor(score)

	s := op.NewScope()
	pbbox := op.Placeholder(s.SubScope("bbox"), tf.Float, op.PlaceholderShape(tf.MakeShape(-1, 4)))
	pscore := op.Placeholder(s.SubScope("score"), tf.Float, op.PlaceholderShape(tf.MakeShape(-1)))

	out := op.NonMaxSuppression(s, pbbox, pscore, op.Const(s.SubScope("max_len"), int32(len(bbox))), op.NonMaxSuppressionIouThreshold(threshold))

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{pbbox: tbbox, pscore: tscore}, []tf.Output{out})
	if err != nil {
		return
	}

	pick := outs[0]
	if pick != nil {
		if idx, ok := pick.Value().([]int32); ok {
			for _, i := range idx {
				nbbox = append(nbbox, bbox[i])
				nreg = append(nreg, reg[i])
				nscore = append(nscore, score[i])
			}
		}
	}

	return
}

func CropResizeImage(img *tf.Tensor, bbox [][]float32, size []int32) (*tf.Tensor, error) {
	h := float32(img.Shape()[1])
	w := float32(img.Shape()[2])
	return cropResizeImage(img, normalizeBbox(bbox, w, h), size, false)
}

func cropResizeImage(img *tf.Tensor, bbox [][]float32, size []int32, normalize bool) (*tf.Tensor, error) {
	tbbox, _ := tf.NewTensor(bbox)

	s := op.NewScope()
	pimg := op.Placeholder(s.SubScope("img"), tf.Float, op.PlaceholderShape(tf.MakeShape(1, -1, -1, 3)))
	pbbox := op.Placeholder(s.SubScope("bbox"), tf.Float, op.PlaceholderShape(tf.MakeShape(-1, 4)))
	ibidx := op.Const(s.SubScope("bidx"), make([]int32, len(bbox)))
	isize := op.Const(s.SubScope("size"), size)

	//	log.Println("cropResize", img.Shape(), ",", tbbox.Shape())

	out := op.CropAndResize(s, pimg, pbbox, ibidx, isize)
	if normalize {
		out = normalizeImage(s, out)
	}

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{pimg: img, pbbox: tbbox}, []tf.Output{out})
	if err != nil {
		return nil, err
	}

	return outs[0], nil
}

func resizeImage(img *tf.Tensor, scale float64) (*tf.Tensor, error) {
	h := int32(math.Ceil(float64(img.Shape()[1]) * scale))
	w := int32(math.Ceil(float64(img.Shape()[2]) * scale))

	s := op.NewScope()
	pimg := op.Placeholder(s, tf.Float, op.PlaceholderShape(tf.MakeShape(1, -1, -1, 3)))

	out := op.ResizeBilinear(s, pimg, op.Const(s.SubScope("size"), []int32{h, w}))
	out = normalizeImage(s, out)

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{pimg: img}, []tf.Output{out})
	if err != nil {
		return nil, err
	}

	return outs[0], nil
}

func normalizeImage(s *op.Scope, input tf.Output) tf.Output {
	out := op.Mul(s, op.Sub(s, input, op.Const(s.SubScope("mean"), float32(127.5))),
		op.Const(s.SubScope("scale"), float32(0.0078125)))
	out = op.Transpose(s, out, op.Const(s.SubScope("perm"), []int32{0, 2, 1, 3}))
	return out
}

func transpose(img *tf.Tensor, perm []int32) (*tf.Tensor, error) {
	s := op.NewScope()
	in := op.Placeholder(s, tf.Float, op.PlaceholderShape(tf.MakeShape(-1, -1, -1, -1)))
	out := op.Transpose(s, in, op.Const(s, perm))

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{in: img}, []tf.Output{out})
	if err != nil {
		return nil, err
	}

	return outs[0], nil
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

func scales(h, w float64, factor, minSize float64) []float64 {
	minl := h
	if minl > w {
		minl = w
	}

	m := 12.0 / minSize
	minl = minl * m

	var scales []float64
	for count := 0; minl > 12.0; {
		scales = append(scales, m*math.Pow(factor, float64(count)))
		minl = minl * factor
		count++
	}

	return scales
}

// RegressMax 回归最大值
func RegressMax(input []float32) (int, error) {
	if len(input) == 0 {
		return 0, errors.New("input list has no elements")
	}
	var output = 0
	MaxElement := input[output]
	for i := range input {
		if input[i] > MaxElement {
			output = i
			MaxElement = input[output]
		}
	}
	return output, nil

}

// Clip 图片裁剪
func Clip(in io.Reader, out io.Writer, x0, y0, x1, y1, quality int) error {
	origin, fm, err := image.Decode(in)
	if err != nil {
		return err
	}

	switch fm {
	case "jpeg":
		img := origin.(*image.YCbCr)
		subImg := img.SubImage(image.Rect(x0, y0, x1, y1)).(*image.YCbCr)
		return jpeg.Encode(out, subImg, &jpeg.Options{Quality: quality})
	case "png":
		switch origin.(type) {
		case *image.NRGBA:
			img := origin.(*image.NRGBA)
			subImg := img.SubImage(image.Rect(x0, y0, x1, y1)).(*image.NRGBA)
			return png.Encode(out, subImg)
		case *image.RGBA:
			img := origin.(*image.RGBA)
			subImg := img.SubImage(image.Rect(x0, y0, x1, y1)).(*image.RGBA)
			return png.Encode(out, subImg)
		}
	case "gif":
		img := origin.(*image.Paletted)
		subImg := img.SubImage(image.Rect(x0, y0, x1, y1)).(*image.Paletted)
		return gif.Encode(out, subImg, &gif.Options{})
	// case "bmp":
	//     img := origin.(*image.RGBA)
	//     subImg := img.SubImage(image.Rect(x0, y0, x1, y1)).(*image.RGBA)
	//     return bmp.Encode(out, subImg)
	default:
		return errors.New("ERROR FORMAT")
	}
	return nil
}

// Scale 缩略图生成
// 入参:
// 规则: 如果width 或 hight其中有一个为0，则大小不变 如果精度为0则精度保持不变
// 矩形坐标系起点是左上
// 返回:error
func Scale(in io.Reader, out io.Writer, width, height, quality int) error {
	origin, fm, err := image.Decode(in)
	if err != nil {
		return err
	}
	if width == 0 || height == 0 {
		width = origin.Bounds().Max.X
		height = origin.Bounds().Max.Y
	}
	if quality <= 0 {
		quality = 100
	}
	// canvas := resize.Thumbnail(uint(width), uint(height), origin, resize.Lanczos3)
	canvas := resize.Resize(uint(width), uint(height), origin, resize.Lanczos3)
	log.Println(origin.Bounds().Max.X)
	log.Println(origin.Bounds().Max.Y)

	//return jpeg.Encode(out, canvas, &jpeg.Options{quality})

	switch fm {
	case "jpeg":
		return jpeg.Encode(out, canvas, &jpeg.Options{Quality: quality})
	case "png":
		return png.Encode(out, canvas)
	case "gif":
		return gif.Encode(out, canvas, &gif.Options{})
	// case "bmp":
	//     return bmp.Encode(out, canvas)
	default:
		return errors.New("ERROR FORMAT")
	}

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
