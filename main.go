package main

import (
	"face-feature/attributes"
	"sync/atomic"
	"time"

	"fmt"
	"io/ioutil"
	"log"

	"dana-tech.com/zhutao/goface/dispatcher"
)

var (
	model string = "./model/estimate.pb"
	image string = "./image/73.jpg"
	i     int64  = 0
)

//JobDetect 定义一个实现Job接口的数据
type JobDetect struct {
	bs []byte //图片数据
}

//Do 定义对数据的处理
func (dt *JobDetect) Do() {
	_, _, err := attributes.GetFaceAttributes(dt.bs)
	if err != nil {
		log.Fatal(err)
		return
	}
	ii := atomic.AddInt64(&i, 1)
	if ii == 10 {
		endCh <- time.Now().UnixNano()
	}
}

var endCh = make(chan int64)

func main() {
	err := attributes.Init(model)
	if err != nil {
		log.Fatal(err)
		return
	}

	img, err := ioutil.ReadFile(image)
	if err != nil {
		log.Fatalf("Init data source error[%v]\n", err)
		return
	}

	maxWorkers := 8
	maxJobs := 1000
	d := dispatcher.NewDispatcher(maxWorkers, maxJobs)
	d.Run()
	start := time.Now().UnixNano()

	go func() {
		for i := 0; i < 10; i++ {
			sc := &JobDetect{
				bs: img,
			}
			d.JobQueue <- sc //数据传进去会被自动执行Do()方法，具体对数据的处理自己在Do()方法中定义
		}
	}()

	end := <-endCh

	fmt.Println("time cost:", ((end - start) / 1e6))
}

func main2() {
	err := attributes.Init(model)
	if err != nil {
		log.Fatal(err)
		return
	}
	start := time.Now().UnixNano()
	bs, err := ioutil.ReadFile(image)

	gender, age, err := attributes.GetFaceAttributes(bs)
	end := time.Now().UnixNano()
	fmt.Println((end-start)/1e6, gender, age)

}
