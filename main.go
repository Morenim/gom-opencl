package main

import (
	"flag"
	"fmt"
	"github.com/Morenim/gom-opencl/bitset"
	"github.com/rainliu/gocl/cl"
	"io/ioutil"
	"log"
	"math/rand"
	"unsafe"
)

var (
	useCPU    bool
	verbosity int
)

func Evaluate(k int, bits bitset.BitSet) (fitness float64) {
	for i := 0; i < bits.Len()/k; i++ {
		t := 0 // number of bits set to 1
		for j := 0; j < k; j++ {
			if bits.Has(i*k + j) {
				t++
			}
		}
		if t == k {
			fitness += float64(t)
		} else {
			fitness += float64(k - t - 1)
		}
	}
	return
}

type byLength [][]int

func (bl byLength) Len() int {
	return len(bl)
}

func (bl byLength) Swap(i, j int) {
	bl[i], bl[j] = bl[j], bl[i]
}

func (bl byLength) Less(i, j int) bool {
	if len(bl[i]) < len(bl[j]) {
		return true
	}
	if len(bl[i]) == len(bl[j]) {
		return bl[i][0] < bl[j][0]
	}
	return false
}

var deviceErrorMap = map[cl.CL_int]string{
	cl.CL_SUCCESS:             "cl: Success",
	cl.CL_DEVICE_NOT_FOUND:    "cl: Device Not Found",
	cl.CL_OUT_OF_HOST_MEMORY:  "cl: Out of Host Memory",
	cl.CL_OUT_OF_RESOURCES:    "cl: Out of Resources",
	cl.CL_INVALID_VALUE:       "cl: Invalid Value",
	cl.CL_INVALID_PLATFORM:    "cl: Invalid Platform",
	cl.CL_INVALID_DEVICE_TYPE: "cl: Invalid Device Type",
}

// Find the first device matching the device type from the list of platforms.
func findDevice(platforms []cl.CL_platform_id, deviceType cl.CL_device_type) (platformID cl.CL_platform_id, deviceID cl.CL_device_id) {

	// Search all platforms for the first device.
	for _, platform := range platforms {

		var numDevices cl.CL_uint

		// Get the number of matching devices for the platform.
		status := cl.CLGetDeviceIDs(
			platform,
			deviceType,
			0,
			nil,
			&numDevices)

		// Check for errors, continue to next platform if no matching device was found.
		switch status {
		case cl.CL_DEVICE_NOT_FOUND:
			fallthrough
		case cl.CL_SUCCESS:
			if numDevices == 0 {
				continue
			}
		default:
			log.Printf("%s", deviceErrorMap[status])
			log.Fatalf("Fatal error: could not retrieve devices for platform %d", platform)
		}

		device := make([]cl.CL_device_id, 1)

		// Select first device matching the device type.
		status = cl.CLGetDeviceIDs(
			platform,
			cl.CL_DEVICE_TYPE_GPU,
			1,
			device,
			nil)

		if status != cl.CL_SUCCESS {
			log.Fatalf("Fatal error: could not retrieve GPU device for platform %d", platform)
		}

		platformID = platform
		deviceID = device[0]
		break
	}

	return
}

func printPlatforms(platforms []cl.CL_platform_id) {

	log.Printf("Debug: found %d platforms:", len(platforms))

	getParam := func(id cl.CL_platform_id, name cl.CL_platform_info) interface{} {
		var numChars cl.CL_size_t
		var info interface{}

		status := cl.CLGetPlatformInfo(id, name, 0, nil, &numChars)
		status = cl.CLGetPlatformInfo(id, name, numChars, &info, nil)

		if status != cl.CL_SUCCESS {
			log.Fatalf("Fatal error: could not retrieve OpenCL platform info for id %d", id)
		}

		return info.(string)
	}

	for _, id := range platforms {
		log.Printf("%s %d", "PlatformID", id)
		log.Printf("\t%-11s: %s", "Name", getParam(id, cl.CL_PLATFORM_NAME))
		log.Printf("\t%-11s: %s", "Vendor", getParam(id, cl.CL_PLATFORM_VENDOR))
		log.Printf("\t%-11s: %s", "Version", getParam(id, cl.CL_PLATFORM_VERSION))
		log.Printf("\t%-11s: %s", "Profile", getParam(id, cl.CL_PLATFORM_PROFILE))
		log.Printf("\t%-11s: %s", "Extensions", getParam(id, cl.CL_PLATFORM_EXTENSIONS))
	}
}

func printGeneration(numGenerations int, pop *Population) {
	fmt.Printf("Generation %d\n", numGenerations)
	fmt.Println("===============")
	for i, solution := range pop.Solutions {
		solution.Fitness = Evaluate(4, solution.Bits)
		fmt.Printf("x_%-2d: %v", i, solution)
	}
	fmt.Println("===============")
	fmt.Println()
}

func flattenIntoSlice(src [][]int, dest []cl.CL_uint) {
	i := 0
	for _, node := range src {
		dest[i] = cl.CL_uint(len(node))
		for _, index := range node {
			dest[i] = cl.CL_uint(index)
			i++
		}
	}
	dest[i] = 0
}

func populationToSlice(pop *Population, dest []cl.CL_uint) {

	for i, solution := range pop.Solutions {
		var raw uint32
		raw = 0

		var j uint32
		for j = 0; j < 32; j++ {
			if solution.Bits.Has(int(j)) {
				raw |= (1 << j)
			}
		}

		dest[i] = cl.CL_uint(raw)
	}
}

func setKernelArg(kernel cl.CL_kernel, pos int, data *cl.CL_mem) {
	status := cl.CLSetKernelArg(
		kernel, cl.CL_uint(pos), cl.CL_size_t(unsafe.Sizeof(data)),
		unsafe.Pointer(data))

	if status != cl.CL_SUCCESS {
		log.Fatalf("Fatal error: could not set arg %d for OpenCL kernel.", pos)
	}
}

func parseCommandLine() {

	flag.BoolVar(&useCPU, "cpu", false, "Whether to use the CPU over the GPU.")

	flag.IntVar(&verbosity, "verbosity", 0, "Verbosity of the output.")

	flag.Parse()
}

func runOpenCL() {

	var status cl.CL_int

	var numPlatforms cl.CL_uint

	//---------------------------------------------------
	// Step 1: Discover and retrieve OpenCL platforms.
	//---------------------------------------------------

	status = cl.CLGetPlatformIDs(0, nil, &numPlatforms)

	platforms := make([]cl.CL_platform_id, numPlatforms)

	status = cl.CLGetPlatformIDs(numPlatforms, platforms, nil)

	if status != cl.CL_SUCCESS {
		log.Fatalf("Fatal error: could not retrieve OpenCL platform IDs.")
	}

	if verbosity >= 2 {
		printPlatforms(platforms)
	}

	//---------------------------------------------------
	// Step 2: Discover and retrieve OpenCL devices.
	//---------------------------------------------------

	var preferredType cl.CL_device_type

	if useCPU {
		preferredType = cl.CL_DEVICE_TYPE_CPU
	} else {
		preferredType = cl.CL_DEVICE_TYPE_GPU
	}

	_, gpuDevice := findDevice(platforms, preferredType)
	gpuDevices := make([]cl.CL_device_id, 1)
	gpuDevices[0] = gpuDevice

	//---------------------------------------------------
	// Step 3: Create an OpenCL context.
	//---------------------------------------------------

	context := cl.CLCreateContext(nil, 1, gpuDevices, nil, nil, &status)

	if status != cl.CL_SUCCESS {
		log.Fatalf("Fatal error: could not create OpenCL context.")
	}

	defer cl.CLReleaseContext(context)

	//---------------------------------------------------
	// Step 3: Create an OpenCL command queue.
	//---------------------------------------------------

	commandQueue := cl.CLCreateCommandQueue(context, gpuDevice, 0, &status)

	if status != cl.CL_SUCCESS {
		log.Fatalf("Fatal error: could not create OpenCL command queue.")
	}

	defer cl.CLReleaseCommandQueue(commandQueue)

	//---------------------------------------------------
	// Step 4: Create OpenCL program and kernel.
	//---------------------------------------------------

	var kernelSource [1][]byte
	var kernelLength [1]cl.CL_size_t
	filename := "kernels/deceptive.cl"

	kernelData, err := ioutil.ReadFile(filename)

	if err != nil {
		log.Fatalf("Could not read the kernel file %s.", filename)
	}

	kernelSource[0] = kernelData
	kernelLength[0] = cl.CL_size_t(len(kernelSource[0]))

	program := cl.CLCreateProgramWithSource(context, 1, kernelSource[:], kernelLength[:], &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not compile an OpenCL kernel from source.")
	}

	status = cl.CLBuildProgram(program, 1, gpuDevices, nil, nil, nil)

	if status != cl.CL_SUCCESS {
		log.Print("Fatal error: could not build OpenCL program.")

		var numChars cl.CL_size_t
		var info interface{}

		status = cl.CLGetProgramBuildInfo(
			program, gpuDevice, cl.CL_PROGRAM_BUILD_LOG,
			0, nil, &numChars)

		status = cl.CLGetProgramBuildInfo(
			program, gpuDevice, cl.CL_PROGRAM_BUILD_LOG,
			numChars, &info, nil)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: could not retrieve OpenCL program build info.")
		}

		log.Fatalf("%s", info.(string))
	}

	kernel := cl.CLCreateKernel(program, []byte("gom"), &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not create OpenCL kernel.")
	}

	//---------------------------------------------------
	// Step 6: Initialize OpenCL memory.
	//---------------------------------------------------

	populationSize := 32

	var size cl.CL_uint
	popSize := cl.CL_size_t(populationSize)
	problemLength := 32
	length := cl.CL_size_t(problemLength)

	pop := NewPopulation(populationSize, problemLength)

	dataSize := cl.CL_size_t(unsafe.Sizeof(size)) * popSize

	populationData := make([]cl.CL_uint, pop.Size())

	offspringData := make([]cl.CL_uint, pop.Size())

	populationBuffer := cl.CLCreateBuffer(
		context, cl.CL_MEM_READ_ONLY, dataSize, nil, &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not allocate an OpenCL memory buffer.")
	}

	// Maximum bound on the number of elements in the LT + node sizes.
	boundSum := (length*length+3*length-2)/2 + (2*length - 1) + 1
	ltSize := cl.CL_size_t(unsafe.Sizeof(length)) * boundSum

	ltData := make([]cl.CL_uint, boundSum)

	ltBuffer := cl.CLCreateBuffer(
		context, cl.CL_MEM_READ_ONLY, ltSize, nil, &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not allocate an OpenCL memory buffer.")
	}

	offspringBuffer := cl.CLCreateBuffer(
		context, cl.CL_MEM_WRITE_ONLY, dataSize, nil, &status)

	if status != cl.CL_SUCCESS {
		log.Fatal("Fatal error: could not allocate an OpenCL memory buffer.")
	}

	//---------------------------------------------------
	// Step 6: Perform GOMEA.
	//---------------------------------------------------

	rand.Seed(2243)

	done := false

	numGenerations := 0

	for !done {

		if verbosity >= 3 {
			printGeneration(numGenerations, pop)
		}

		//---------------------------------------------------
		// Step 5: Initialize the LTGA.
		//---------------------------------------------------
		freqs := Frequencies(pop)

		lt := LinkageTree(pop, freqs)

		// Store a flattened version of the linkage tree in memory.
		flattenIntoSlice(lt, ltData)

		// Upload FOS.
		status = cl.CLEnqueueWriteBuffer(
			commandQueue, ltBuffer, cl.CL_TRUE, 0,
			ltSize, unsafe.Pointer(&ltData[0]), 0, nil, nil)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: could not write data to an OpenCL memory buffer.")
		}

		populationToSlice(pop, populationData)

		status = cl.CLEnqueueWriteBuffer(
			commandQueue, populationBuffer, cl.CL_TRUE, 0,
			dataSize, unsafe.Pointer(&populationData[0]), 0, nil, nil)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: could not write data to an OpenCL memory buffer.")
		}

		setKernelArg(kernel, 0, &populationBuffer)

		clSize := cl.CL_uint(populationSize)
		status := cl.CLSetKernelArg(
			kernel, 1, cl.CL_size_t(unsafe.Sizeof(clSize)),
			unsafe.Pointer(&popSize))

		if status != cl.CL_SUCCESS {
			log.Printf("%v", cl.ERROR_CODES_STRINGS[-status])
			log.Fatalf("Fatal error: could not set arg %d for OpenCL kernel.", 1)
		}

		setKernelArg(kernel, 2, &ltBuffer)

		setKernelArg(kernel, 3, &offspringBuffer)

		var globalWorkSize [1]cl.CL_size_t
		globalWorkSize[0] = cl.CL_size_t(pop.Size())

		//---------------------------------------------------
		// Step 7: Perform GOM crossover.
		//---------------------------------------------------

		status = cl.CLEnqueueNDRangeKernel(
			commandQueue, kernel, 1, nil, globalWorkSize[:],
			nil, 0, nil, nil)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: could not enqueue OpenCL kernel.")
		}

		status = cl.CLFinish(commandQueue)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: could not finish command queue.")
		}

		cl.CLEnqueueReadBuffer(
			commandQueue, offspringBuffer, cl.CL_TRUE, 0,
			dataSize, unsafe.Pointer(&offspringData[0]), 0, nil, nil)

		if status != cl.CL_SUCCESS {
			log.Fatal("Fatal error: reading a buffer failed.")
		}

		for i, offspring := range offspringData {
			pop.Solutions[i].Bits, _ = bitset.FromString(fmt.Sprintf("%032b", uint(offspring)))
			pop.Solutions[i].Fitness = Evaluate(4, pop.Solutions[i].Bits)
		}

		numGenerations++

		// TODO: Termination Criterion
		if numGenerations == 10 {
			done = true
		}
	}
}

func main() {

	parseCommandLine()

	runOpenCL()
}
